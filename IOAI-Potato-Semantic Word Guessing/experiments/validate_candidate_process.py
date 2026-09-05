from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import select
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORDS = json.loads((ROOT / "dataset/vocabulary.json").read_text())
INDEX = {word.casefold(): index for index, word in enumerate(WORDS)}
LAMP = INDEX["lamp"]
POTATO = INDEX["potato"]


def selected_indices(salt: str, count: int) -> list[int]:
    return sorted(
        range(len(WORDS)),
        key=lambda index: hashlib.sha256(
            (salt + "\0" + WORDS[index]).encode()
        ).digest(),
    )[:count]


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.sqrt(np.einsum("ij,ij->i", values, values))
    return np.asarray(values / norms[:, None], dtype=np.float32)


def normalized_embeddings() -> np.ndarray:
    embeddings = np.load(
        ROOT / "dataset/public_embeddings.npy",
        mmap_mode="r",
    )
    return normalize_rows(embeddings)


def stress_fit_indices() -> np.ndarray:
    selected = []
    for index, word in enumerate(WORDS):
        digest = hashlib.sha256(
            ("stress-secret-v1\0" + word).encode()
        ).digest()
        if int.from_bytes(digest[:8], "big") % 16 < 8:
            selected.append(index)
    return np.asarray(selected, dtype=np.int64)


def transformed_oracle(
    name: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    vectors = normalized_embeddings()
    if name == "raw":
        return vectors, None
    fit = stress_fit_indices()
    mean = vectors[fit].mean(axis=0)
    if name in {"half_center", "center"}:
        alpha = 0.5 if name == "half_center" else 1.0
        return normalize_rows(vectors - alpha * mean), None
    if name == "mask60":
        order = sorted(
            range(vectors.shape[1]),
            key=lambda dimension: hashlib.sha256(
                f"crosscut-blind-v1\0{dimension}".encode()
            ).digest(),
        )
        keep = np.asarray(
            order[: round(0.60 * vectors.shape[1])],
            dtype=np.int64,
        )
        return normalize_rows(vectors[:, keep]), None
    if name == "spectral16":
        centered_fit = vectors[fit] - mean
        _, _, right = np.linalg.svd(
            centered_fit,
            full_matrices=False,
        )
        basis = right[:16]
        signs = np.asarray(
            [
                1.0
                if hashlib.sha256(
                    f"spectral-blind-v1\0{index}".encode()
                ).digest()[0]
                % 2
                else -1.0
                for index in range(16)
            ],
            dtype=np.float32,
        )
        centered = vectors - mean
        coefficients = centered @ basis.T
        factors = np.exp(0.20 * signs) - 1.0
        transformed = (
            mean
            + centered
            + (coefficients * factors) @ basis
        )
        return normalize_rows(transformed), None
    similarities = np.asarray(vectors @ vectors.T, dtype=np.float32)
    if name == "csls":
        neighbors = 24
        top = np.partition(
            similarities,
            len(WORDS) - neighbors - 1,
            axis=1,
        )[:, len(WORDS) - neighbors - 1 :]
        diagonal = np.diag(similarities)
        density = (top.sum(axis=1) - diagonal) / neighbors
        return vectors, np.asarray(density, dtype=np.float32)
    nearest = np.argpartition(
        similarities,
        len(WORDS) - 17,
        axis=1,
    )[:, len(WORDS) - 17 :]
    means = np.empty_like(vectors)
    for index in range(len(WORDS)):
        neighbors = nearest[index]
        neighbors = neighbors[neighbors != index]
        if len(neighbors) > 16:
            neighbor_scores = similarities[index, neighbors]
            neighbors = neighbors[
                np.argsort(neighbor_scores)[-16:]
            ]
        means[index] = vectors[neighbors].mean(axis=0)
    if name == "smooth25":
        transformed = 0.75 * vectors + 0.25 * means
    elif name == "sharpen16":
        transformed = 1.16 * vectors - 0.16 * means
    else:
        raise ValueError(name)
    return normalize_rows(transformed), None


def read_stable_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"not a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    data = b"".join(chunks)
    if identity_before != identity_after or len(data) != after.st_size:
        raise RuntimeError(f"file changed while reading: {path}")
    return data


def send(process: subprocess.Popen, message: dict) -> None:
    if process.stdin is None:
        raise RuntimeError("child stdin unavailable")
    process.stdin.write((json.dumps(message) + "\n").encode())
    process.stdin.flush()


def reject_unsolicited_output(
    process: subprocess.Popen,
    buffer: bytearray,
) -> None:
    if buffer:
        raise RuntimeError("child emitted multiple or partial response lines")
    if process.stdout is None:
        raise RuntimeError("child stdout unavailable")
    ready, _, _ = select.select([process.stdout.fileno()], [], [], 0.0)
    if ready:
        chunk = os.read(process.stdout.fileno(), 1024)
        if chunk:
            raise RuntimeError("child emitted unsolicited stdout")
        if process.poll() is not None:
            raise RuntimeError("child stopped unexpectedly")


def read_line(
    process: subprocess.Popen,
    buffer: bytearray,
    deadline: float,
) -> str:
    if process.stdout is None:
        raise RuntimeError("child stdout unavailable")
    descriptor = process.stdout.fileno()
    while True:
        newline = buffer.find(b"\n")
        if newline >= 0:
            line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            try:
                return line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise RuntimeError("child response is not UTF-8") from error
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("child response timed out")
        ready, _, _ = select.select([descriptor], [], [], remaining)
        if not ready:
            raise TimeoutError("child response timed out")
        chunk = os.read(descriptor, 1024)
        if not chunk:
            raise RuntimeError("child stopped before responding")
        buffer.extend(chunk)
        if len(buffer) > 4096:
            raise RuntimeError("child response exceeded 4096 bytes")


def stop(process: subprocess.Popen) -> None:
    if process.stdin is not None and not process.stdin.closed:
        process.stdin.close()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


def systemd_environment() -> dict[str, str]:
    environment = os.environ.copy()
    runtime_directory = f"/run/user/{os.getuid()}"
    environment["XDG_RUNTIME_DIR"] = runtime_directory
    environment["DBUS_SESSION_BUS_ADDRESS"] = (
        f"unix:path={runtime_directory}/bus"
    )
    return environment


def systemctl(
    arguments: list[str],
    environment: dict[str, str],
    check: bool = False,
) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        ["systemctl", "--user", *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        check=False,
        timeout=5,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.decode(errors="replace").strip()
            or "systemctl command failed"
        )
    return completed


def systemctl_show(
    unit: str,
    environment: dict[str, str],
    properties: list[str],
) -> dict[str, str]:
    completed = systemctl(
        [
            "show",
            unit,
            "--no-pager",
            *[f"--property={name}" for name in properties],
        ],
        environment,
        check=True,
    )
    values = {}
    for line in completed.stdout.decode().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    missing = set(properties) - set(values)
    if missing:
        raise RuntimeError(
            f"missing systemd properties: {sorted(missing)}"
        )
    return values


def expected_systemd_duration(seconds: float) -> str:
    if seconds == 60.0:
        return "1min"
    if seconds.is_integer():
        return f"{int(seconds)}s"
    return f"{seconds}s"


def read_cgroup_snapshot(path: Path) -> dict:
    names = [
        "memory.current",
        "memory.peak",
        "memory.max",
        "memory.swap.current",
        "memory.swap.peak",
        "memory.swap.max",
    ]
    values = {}
    for name in names:
        text = (path / name).read_text().strip()
        if not text.isdigit():
            raise RuntimeError(f"invalid cgroup value in {name}")
        values[name.replace(".", "_")] = int(text)
    events = {}
    for line in (path / "memory.events").read_text().splitlines():
        key, value = line.split()
        events[key] = int(value)
    for required in ["oom", "oom_kill", "oom_group_kill"]:
        if required not in events:
            raise RuntimeError(
                f"missing cgroup memory event {required}"
            )
    pids = {
        int(line)
        for line in (path / "cgroup.procs").read_text().splitlines()
        if line
    }
    return {**values, "memory_events": events, "pids": pids}


def systemd_service_info(
    unit: str,
    environment: dict[str, str],
    deadline: float,
    hard_runtime_seconds: float,
) -> tuple[int, dict]:
    names = [
        "Type",
        "TimeoutStopUSec",
        "RuntimeMaxUSec",
        "MainPID",
        "OOMPolicy",
        "ControlGroup",
        "MemoryAccounting",
        "MemoryMax",
        "MemorySwapMax",
        "KillMode",
    ]
    while time.monotonic() < deadline:
        try:
            properties = systemctl_show(unit, environment, names)
        except RuntimeError:
            time.sleep(0.01)
            continue
        text = properties["MainPID"]
        if text.isdigit() and int(text) > 0:
            break
        time.sleep(0.01)
    else:
        raise RuntimeError("hard-limit solution process did not start")
    expected = {
        "Type": "exec",
        "TimeoutStopUSec": "1s",
        "RuntimeMaxUSec": expected_systemd_duration(
            hard_runtime_seconds
        ),
        "OOMPolicy": "kill",
        "MemoryAccounting": "yes",
        "MemoryMax": str(64 * 1024 * 1024),
        "MemorySwapMax": "0",
        "KillMode": "control-group",
    }
    mismatches = {
        name: {
            "expected": value,
            "actual": properties.get(name),
        }
        for name, value in expected.items()
        if properties.get(name) != value
    }
    if mismatches:
        raise RuntimeError(f"hard-limit property mismatch: {mismatches}")
    main_pid = int(properties["MainPID"])
    control_group = properties["ControlGroup"]
    if not control_group.startswith("/"):
        raise RuntimeError("invalid systemd control group")
    process_groups = Path(f"/proc/{main_pid}/cgroup").read_text().splitlines()
    if f"0::{control_group}" not in process_groups:
        raise RuntimeError("solution PID is outside the transient cgroup")
    cgroup_path = Path("/sys/fs/cgroup") / control_group.lstrip("/")
    snapshot = read_cgroup_snapshot(cgroup_path)
    if (
        snapshot["memory_max"] != 64 * 1024 * 1024
        or snapshot["memory_swap_max"] != 0
        or main_pid not in snapshot["pids"]
    ):
        raise RuntimeError("cgroup files do not match hard limits")
    return main_pid, {
        "unit": unit,
        "control_group": control_group,
        "cgroup_path": cgroup_path,
        "properties": properties,
        "limits_verified": True,
        "main_pid_member_at_start": True,
        "hard_runtime_seconds": hard_runtime_seconds,
    }


def resource_monitor(
    pid: int,
    cgroup_path: Path | None,
    done: threading.Event,
    result: dict,
) -> None:
    status = Path(f"/proc/{pid}/status")
    peak_rss = 0
    peak_hwm = 0
    cgroup_peak = 0
    swap_peak = 0
    swap_current = 0
    samples = 0
    member_seen = False
    events = {}
    while not done.is_set():
        try:
            for line in status.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    peak_rss = max(peak_rss, int(line.split()[1]))
                elif line.startswith("VmHWM:"):
                    peak_hwm = max(peak_hwm, int(line.split()[1]))
        except (FileNotFoundError, ProcessLookupError):
            pass
        if cgroup_path is not None:
            try:
                snapshot = read_cgroup_snapshot(cgroup_path)
                samples += 1
                cgroup_peak = max(
                    cgroup_peak,
                    snapshot["memory_peak"],
                )
                swap_peak = max(
                    swap_peak,
                    snapshot["memory_swap_peak"],
                )
                swap_current = max(
                    swap_current,
                    snapshot["memory_swap_current"],
                )
                member_seen = member_seen or pid in snapshot["pids"]
                events = snapshot["memory_events"]
            except FileNotFoundError:
                break
        done.wait(0.01)
    result.update(
        {
            "vmrss_kib": peak_rss,
            "vmhwm_kib": peak_hwm,
            "cgroup_memory_peak_bytes": cgroup_peak,
            "cgroup_memory_peak_kib": (
                (cgroup_peak + 1023) // 1024 if cgroup_peak else 0
            ),
            "cgroup_swap_peak_bytes": swap_peak,
            "cgroup_swap_current_max_bytes": swap_current,
            "cgroup_memory_events": events,
            "cgroup_samples": samples,
            "main_pid_seen_in_cgroup": member_seen,
        }
    )


def start_solution(
    script: Path,
    environment: dict[str, str],
    hard_limits: bool,
    stderr_file,
    hard_runtime_seconds: float = 60.0,
) -> tuple[
    subprocess.Popen,
    int,
    dict | None,
    dict[str, str],
]:
    if not hard_limits:
        process = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=ROOT,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            start_new_session=True,
        )
        return process, process.pid, None, environment
    manager_environment = systemd_environment()
    unit = f"potato-resource-{os.getpid()}-{time.monotonic_ns()}.service"
    service_environment = [
        f"--setenv={name}={environment[name]}"
        for name in [
            "PYTHONDONTWRITEBYTECODE",
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]
    ]
    process = subprocess.Popen(
        [
            "systemd-run",
            "--user",
            "--pipe",
            "--wait",
            "--quiet",
            f"--unit={unit}",
            "--property=Type=exec",
            "--property=MemoryAccounting=yes",
            "--property=MemoryMax=64M",
            "--property=MemorySwapMax=0",
            "--property=OOMPolicy=kill",
            "--property=KillMode=control-group",
            f"--property=RuntimeMaxSec={hard_runtime_seconds}s",
            "--property=TimeoutStopSec=1s",
            f"--working-directory={ROOT}",
            *service_environment,
            sys.executable,
            str(script),
        ],
        cwd=ROOT,
        env=manager_environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_file,
        start_new_session=True,
    )
    try:
        main_pid, context = systemd_service_info(
            unit,
            manager_environment,
            time.monotonic() + 10.0,
            hard_runtime_seconds,
        )
    except Exception:
        systemctl(
            ["kill", "--kill-whom=all", "--signal=SIGKILL", unit],
            manager_environment,
        )
        systemctl(["stop", unit], manager_environment)
        stop(process)
        raise
    return process, main_pid, context, manager_environment


def stop_systemd_unit(
    context: dict | None,
    environment: dict[str, str],
) -> bool:
    if context is None:
        return True
    unit = context["unit"]
    cgroup_path = context["cgroup_path"]
    systemctl(["stop", unit], environment)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            pids = [
                line
                for line in (cgroup_path / "cgroup.procs")
                .read_text()
                .splitlines()
                if line
            ]
        except FileNotFoundError:
            pids = []
        if not pids:
            break
        systemctl(
            ["kill", "--kill-whom=all", "--signal=SIGKILL", unit],
            environment,
        )
        time.sleep(0.02)
    systemctl(["reset-failed", unit], environment)
    return not pids


def score(turn: int | None) -> float:
    if turn is None:
        return 0.0
    return 1.0 - 0.02 * max(0, turn - 10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--salt", default="candidate-process-v1")
    parser.add_argument("--time-limit", type=float, default=90.0)
    parser.add_argument("--force-max-turns", action="store_true")
    parser.add_argument("--hard-limits", action="store_true")
    parser.add_argument(
        "--oracle",
        choices=[
            "raw",
            "half_center",
            "center",
            "mask60",
            "spectral16",
            "smooth25",
            "sharpen16",
            "csls",
        ],
        default="raw",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if args.games <= 0 or args.games > len(WORDS):
        parser.error("games must be between 1 and vocabulary size")
    if args.time_limit <= 0.0:
        parser.error("time-limit must be positive")

    notebook_path = args.notebook.resolve()
    notebook_bytes = read_stable_bytes(notebook_path)
    notebook_sha256 = hashlib.sha256(notebook_bytes).hexdigest()
    if (
        args.expected_sha256 is not None
        and notebook_sha256 != args.expected_sha256
    ):
        raise RuntimeError(
            "notebook hash does not match --expected-sha256"
        )
    vectors, density = transformed_oracle(args.oracle)
    secrets = selected_indices(args.salt, args.games)
    started = time.monotonic()
    workspace = tempfile.TemporaryDirectory(prefix="potato-process-")
    snapshot_notebook = Path(workspace.name) / "snapshot.ipynb"
    snapshot_notebook.write_bytes(notebook_bytes)
    converted = Path(workspace.name) / "solution.py"
    conversion_started = time.monotonic()
    conversion = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "python",
            str(snapshot_notebook),
            "--output",
            "solution",
            "--output-dir",
            workspace.name,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    conversion_elapsed = time.monotonic() - conversion_started
    if conversion.returncode != 0 or not converted.is_file():
        workspace.cleanup()
        raise RuntimeError(conversion.stderr.decode(errors="replace"))
    converted_sha256 = hashlib.sha256(
        read_stable_bytes(converted)
    ).hexdigest()

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    solution_started = time.monotonic()
    process = None
    context = None
    process_environment = environment
    memory = {}
    monitor_done = threading.Event()
    monitor = None
    buffer = bytearray()
    turns = []
    responses = 0
    stderr_text = ""
    trailing_stdout = b""
    cleanup_verified = True
    stderr_file = tempfile.TemporaryFile()
    try:
        (
            process,
            monitored_pid,
            context,
            process_environment,
        ) = start_solution(
            converted,
            environment,
            args.hard_limits,
            stderr_file,
        )
        monitor = threading.Thread(
            target=resource_monitor,
            args=(
                monitored_pid,
                (
                    context["cgroup_path"]
                    if context is not None
                    else None
                ),
                monitor_done,
                memory,
            ),
            daemon=True,
        )
        monitor.start()
        effective_time_limit = min(
            args.time_limit,
            60.0 if args.hard_limits else args.time_limit,
        )
        deadline = solution_started + effective_time_limit
        for secret in secrets:
            reject_unsolicited_output(process, buffer)
            send(process, {"event": "new_game"})
            first = LAMP
            second = POTATO
            found = None
            for turn in range(1, 31):
                reject_unsolicited_output(process, buffer)
                first_similarity = float(
                    vectors[secret] @ vectors[first]
                )
                second_similarity = float(
                    vectors[secret] @ vectors[second]
                )
                if density is not None:
                    first_similarity = (
                        2.0 * first_similarity - float(density[first])
                    )
                    second_similarity = (
                        2.0 * second_similarity - float(density[second])
                    )
                difference = first_similarity - second_similarity
                if abs(difference) <= 1e-12:
                    winner = first
                    verdict = "same"
                elif difference > 0.0:
                    winner = first
                    verdict = "first"
                else:
                    winner = second
                    verdict = "second"
                send(
                    process,
                    {
                        "turn": turn,
                        "winner_word": WORDS[winner],
                        "verdict": verdict,
                        "word1": WORDS[first],
                        "word2": WORDS[second],
                    },
                )
                response = json.loads(
                    read_line(process, buffer, deadline)
                )
                reject_unsolicited_output(process, buffer)
                responses += 1
                if (
                    not isinstance(response, dict)
                    or set(response) != {"new_word"}
                    or not isinstance(response["new_word"], str)
                ):
                    raise RuntimeError("invalid response object")
                proposal = INDEX.get(response["new_word"].casefold())
                if proposal is None:
                    raise RuntimeError("response word outside vocabulary")
                if proposal == secret and found is None:
                    found = turn
                    if not args.force_max_turns:
                        reject_unsolicited_output(process, buffer)
                        send(process, {"status": "win"})
                        break
                first = winner
                second = proposal
            if args.force_max_turns:
                reject_unsolicited_output(process, buffer)
                send(
                    process,
                    {"status": "win" if found is not None else "loss"},
                )
            elif found is None:
                reject_unsolicited_output(process, buffer)
                send(process, {"status": "loss"})
            turns.append(found)
        reject_unsolicited_output(process, buffer)
        send(process, {"event": "done"})
    finally:
        if process is not None:
            stop(process)
        monitor_done.set()
        if monitor is not None:
            monitor.join(timeout=2)
        if context is not None:
            cleanup_verified = stop_systemd_unit(
                context,
                process_environment,
            )
        if process is not None and process.stdout is not None:
            trailing_stdout = bytes(buffer) + process.stdout.read()
        stderr_file.seek(0)
        stderr_text = stderr_file.read().decode(errors="replace")
        stderr_file.close()
        workspace.cleanup()

    solution_elapsed = time.monotonic() - solution_started
    if process is None:
        raise RuntimeError("solution process was not created")
    if process.returncode != 0:
        raise RuntimeError(
            f"solution process exited with {process.returncode}: "
            f"{stderr_text.strip()}"
        )
    if trailing_stdout:
        raise RuntimeError("solution emitted trailing stdout")
    if stderr_text:
        raise RuntimeError(
            f"solution emitted stderr: {stderr_text.strip()}"
        )
    if read_stable_bytes(notebook_path) != notebook_bytes:
        raise RuntimeError("notebook changed during validation")
    hard_limit_checks = {}
    if args.hard_limits:
        events = memory.get("cgroup_memory_events", {})
        hard_limit_checks = {
            "unit_properties": (
                context is not None
                and context.get("limits_verified") is True
            ),
            "main_pid_membership": (
                context is not None
                and context.get("main_pid_member_at_start") is True
                and memory.get("main_pid_seen_in_cgroup") is True
            ),
            "samples_positive": memory.get("cgroup_samples", 0) > 0,
            "memory_peak_positive": (
                memory.get("cgroup_memory_peak_bytes", 0) > 0
            ),
            "memory_peak_within_limit": (
                0
                < memory.get("cgroup_memory_peak_bytes", 0)
                <= 64 * 1024 * 1024
            ),
            "swap_current_zero": (
                memory.get("cgroup_swap_current_max_bytes", -1) == 0
            ),
            "swap_peak_zero": (
                memory.get("cgroup_swap_peak_bytes", -1) == 0
            ),
            "oom_zero": events.get("oom") == 0,
            "oom_kill_zero": events.get("oom_kill") == 0,
            "oom_group_kill_zero": (
                events.get("oom_group_kill") == 0
            ),
            "cleanup": cleanup_verified,
            "wall_time": solution_elapsed <= 60.0,
        }
        if not all(hard_limit_checks.values()):
            raise RuntimeError(
                f"hard-limit verification failed: {hard_limit_checks}"
            )
    total_elapsed = time.monotonic() - started
    child_rusage_kib = resource.getrusage(
        resource.RUSAGE_CHILDREN
    ).ru_maxrss
    wins = sum(turn is not None for turn in turns)
    result = {
        "validator_sha256": hashlib.sha256(
            read_stable_bytes(Path(__file__))
        ).hexdigest(),
        "notebook": str(args.notebook),
        "notebook_sha256": notebook_sha256,
        "converted_sha256": converted_sha256,
        "salt": args.salt,
        "games": args.games,
        "force_max_turns": args.force_max_turns,
        "hard_limits": args.hard_limits,
        "hard_limit_checks": hard_limit_checks,
        "oracle": args.oracle,
        "responses": responses,
        "wins": wins,
        "score": round(
            100.0 * sum(score(turn) for turn in turns) / len(turns),
            4,
        ),
        "turns": turns,
        "conversion_seconds": round(conversion_elapsed, 4),
        "solution_seconds": round(solution_elapsed, 4),
        "total_seconds": round(total_elapsed, 4),
        "vmrss_kib": memory.get("vmrss_kib", 0),
        "vmhwm_kib": memory.get("vmhwm_kib", 0),
        "cgroup_memory_peak_bytes": memory.get(
            "cgroup_memory_peak_bytes",
            0,
        ),
        "cgroup_memory_peak_kib": memory.get(
            "cgroup_memory_peak_kib",
            0,
        ),
        "cgroup_swap_peak_bytes": memory.get(
            "cgroup_swap_peak_bytes",
            0,
        ),
        "cgroup_swap_current_max_bytes": memory.get(
            "cgroup_swap_current_max_bytes",
            0,
        ),
        "cgroup_memory_events": memory.get(
            "cgroup_memory_events",
            {},
        ),
        "cgroup_samples": memory.get("cgroup_samples", 0),
        "cgroup": (
            {
                "control_group": context["control_group"],
                "properties": context["properties"],
            }
            if context is not None
            else None
        ),
        "rusage_children_kib": child_rusage_kib,
        "stderr": stderr_text,
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(
            f".{args.output.name}.tmp-{os.getpid()}"
        )
        temporary.write_text(text)
        os.replace(temporary, args.output)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
