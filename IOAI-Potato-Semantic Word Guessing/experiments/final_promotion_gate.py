from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
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
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"
FINAL_GATE = ROOT / "experiments/final_gate"
MARKER_PATH = FINAL_GATE / "opened-v2.json"
OUTPUT_PATH = FINAL_GATE / "result-v2.json"
VOCABULARY = ROOT / "dataset/vocabulary.json"
EMBEDDINGS = ROOT / "dataset/public_embeddings.npy"
PUBLIC_TEST = ROOT / "dataset/test_public.json"
ROOT_INCUMBENT = ROOT / "solution.ipynb"
ROOT_INCUMBENT_ZIP = ROOT / "solution.zip"
BASELINE = ROOT / "baselines/score_57_02/solution.ipynb"
CANDIDATE = ROOT / "experiments/candidates/consistency_base/solution.ipynb"
CANDIDATE_SOURCE = (
    ROOT / "experiments/candidates/consistency_base/solution.py"
)
VALIDATOR = ROOT / "experiments/validate_candidate_process.py"
EXPECTED_BASELINE = (
    "2cc93b01302c0057b159780f7e0e41715d3c1cd8902487712e56ce267492c079"
)
EXPECTED_INCUMBENT_ZIP = (
    "4670447bd8d816c537f31021a23d27f136e596ed9ee77fc5d5b3ac043c863bfa"
)
EXPECTED_CANDIDATE = (
    "7f6e901b2001f668f3bc5007e643b954c57fccefd03c014fc6599d138482caac"
)
EXPECTED_CANDIDATE_SOURCE = (
    "24d048df814ed77a1151f6ed532c3a17eb71e0c6a56b06f0c2d631d7b2a73b60"
)
EXPECTED_VALIDATOR = (
    "51a43808a134b146bad7e5248c454b04d2bc3295eb47438e61b40b8d9174c112"
)
EXPECTED_CONVERTED_CANDIDATE = (
    "d04eefe3a5ea8936d6a91cb5b0d41fd1b16d2d132f1531a37552eeac15bdbe1b"
)
GATE_SALT = "ioai26-potato-final-promotion-gate-20260827-v2"
MASK_SALT = "ioai26-potato-final-gate-mask50-v2"
SPECTRAL_SALT = "ioai26-potato-final-gate-spectral24-v2"
BOOTSTRAP_SALT = "ioai26-potato-final-promotion-bootstrap-v2"
CELL_NAMES = [
    "full_raw",
    "full_center75",
    "full_mask50",
    "full_spectral24",
    "held_raw",
    "held_mix60",
    "held_center75",
    "held_csls48",
]
CELL_SIZE = 32
GATE_SIZE = CELL_SIZE * len(CELL_NAMES)
EXPECTED_USED_COUNT = 1301
EXPECTED_REMAINING_COUNT = 301
EXPECTED_RESERVE_COUNT = 45
MANUAL_USED_SALTS = [
    ("candidate-equivalence-v1", 32),
    ("quantization-ablation-v1", 24),
    ("candidate-quantized-equivalence-v1", 32),
    ("consistency-switch-dev-v1", 64),
    ("consistency-quant-guard-v1", 96),
    ("consistency-candidate-equivalence-v1", 48),
    ("consistency-dedup-equivalence-v1", 64),
    ("code-equivalence-sealed-v1", 120),
    ("consistency-resource-audit-final-v1", 120),
    ("consistency-resource-current-patch-v1", 120),
    ("code-audit-games-v1", 96),
    ("code-audit-pairs-v1", 128),
    ("memory-cold-repeat-a", 1),
    ("memory-cold-repeat-b", 1),
]
RESOURCE_RESULTS = [
    f"consistency-resource-hardlimit-repeat-{index}.json"
    for index in range(1, 6)
]
EXPECTED_HARD_LIMIT_CHECKS = {
    "unit_properties",
    "main_pid_membership",
    "samples_positive",
    "memory_peak_positive",
    "memory_peak_within_limit",
    "swap_current_zero",
    "swap_peak_zero",
    "oom_zero",
    "oom_kill_zero",
    "oom_group_kill_zero",
    "cleanup",
    "wall_time",
}
EXPECTED_CGROUP_PROPERTIES = {
    "Type": "exec",
    "TimeoutStopUSec": "1s",
    "RuntimeMaxUSec": "1min",
    "OOMPolicy": "kill",
    "MemoryAccounting": "yes",
    "MemoryMax": str(64 * 1024 * 1024),
    "MemorySwapMax": "0",
    "KillMode": "control-group",
}
THRESHOLDS = {
    "overall_score_delta_min": 2.0,
    "bootstrap_lower99_strictly_positive": True,
    "overall_win_delta_min": 8,
    "baseline_win_rate_max": 95.0,
    "cell_score_delta_min": -1.0,
    "cell_win_delta_min": -1,
    "raw_harms_max": 0,
    "resource_runs_required": 5,
    "resource_games_required": 120,
    "resource_responses_required": 3600,
    "resource_solution_seconds_max": 45.0,
    "resource_total_seconds_max": 50.0,
    "resource_vmhwm_kib_max": 58000,
    "resource_cgroup_memory_peak_bytes_max": 64 * 1024 * 1024,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def commitment(value: object) -> str:
    return sha256_bytes(canonical_bytes(value))


def read_stable_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"Not a regular file: {path}")
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
        raise RuntimeError(f"File changed while reading: {path}")
    return data


def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value: {value}")


def strict_json_bytes(data: bytes, source: str) -> object:
    try:
        text = data.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"Malformed JSON in {source}: {error}") from error


VOCABULARY_BYTES = read_stable_bytes(VOCABULARY)
WORDS_OBJECT = strict_json_bytes(VOCABULARY_BYTES, str(VOCABULARY))
if not isinstance(WORDS_OBJECT, list) or not all(
    isinstance(word, str) and word for word in WORDS_OBJECT
):
    raise RuntimeError("Invalid vocabulary")
WORDS = list(WORDS_OBJECT)
if len(set(word.casefold() for word in WORDS)) != len(WORDS):
    raise RuntimeError("Vocabulary is not case-insensitively unique")
INDEX = {word.casefold(): index for index, word in enumerate(WORDS)}
LAMP = INDEX["lamp"]
POTATO = INDEX["potato"]


def exact_int(
    value: object,
    label: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise RuntimeError(f"{label} is out of range")
    return value


def finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} must be finite")
    return result


def selected_indices(salt: str, count: int) -> list[int]:
    count = exact_int(count, "selection count", 1, len(WORDS))
    return sorted(
        range(len(WORDS)),
        key=lambda index: hashlib.sha256(
            (salt + "\0" + WORDS[index]).encode()
        ).digest(),
    )[:count]


def validate_turns(value: object, games: int, label: str) -> None:
    if not isinstance(value, list) or len(value) != games:
        raise RuntimeError(f"{label} must contain exactly {games} turns")
    for turn in value:
        if turn is None:
            continue
        exact_int(turn, label, 1, 30)


def validate_indices(
    value: object,
    expected_length: int,
    label: str,
) -> list[int]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise RuntimeError(
            f"{label} must contain exactly {expected_length} indices"
        )
    indices = [
        exact_int(item, label, 0, len(WORDS) - 1)
        for item in value
    ]
    if len(set(indices)) != len(indices):
        raise RuntimeError(f"{label} contains duplicate indices")
    return indices


def search_result_selection_count(data: dict, source: str) -> int:
    count = exact_int(data.get("count"), f"{source} count", 1)
    mode = data.get("secret_mode", "shared")
    if mode not in {"shared", "cell_disjoint"}:
        raise RuntimeError(f"{source} has invalid secret_mode")
    if mode == "shared":
        return exact_int(count, f"{source} selection count", 1, len(WORDS))
    folds = data.get("folds")
    oracles = data.get("oracles")
    if not isinstance(folds, list) or not folds:
        raise RuntimeError(f"{source} has invalid folds")
    if not isinstance(oracles, list) or not oracles:
        raise RuntimeError(f"{source} has invalid oracles")
    needed = count * len(folds) * len(oracles)
    return exact_int(
        needed,
        f"{source} disjoint selection count",
        1,
        len(WORDS),
    )


def used_indices_from_result(data: object, source: str) -> set[int]:
    if not isinstance(data, dict):
        raise RuntimeError(f"{source} must contain a JSON object")
    salt = data.get("salt")
    if not isinstance(salt, str) or not salt:
        raise RuntimeError(f"{source} has no valid salt")
    rows = data.get("rows")
    if rows is not None:
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"{source} has invalid rows")
        if not all(isinstance(row, dict) for row in rows):
            raise RuntimeError(f"{source} has a non-object row")
        explicit = ["secret_indices" in row for row in rows]
        if any(explicit):
            if not all(explicit):
                raise RuntimeError(
                    f"{source} has partial explicit provenance"
                )
            observed: set[int] = set()
            for row_index, row in enumerate(rows):
                games = exact_int(
                    row.get("games"),
                    f"{source} row {row_index} games",
                    1,
                    len(WORDS),
                )
                indices = validate_indices(
                    row.get("secret_indices"),
                    games,
                    f"{source} row {row_index} secret_indices",
                )
                observed.update(indices)
                if "turns" in row:
                    validate_turns(
                        row["turns"],
                        games,
                        f"{source} row {row_index} turns",
                    )
            needed = search_result_selection_count(data, source)
            expected = set(selected_indices(salt, needed))
            if observed != expected:
                raise RuntimeError(
                    f"{source} explicit provenance does not match its salt"
                )
            return observed
    if "games" in data:
        games = exact_int(
            data["games"],
            f"{source} games",
            1,
            len(WORDS),
        )
        if "turns" in data:
            validate_turns(data["turns"], games, f"{source} turns")
        if "responses" in data:
            exact_int(
                data["responses"],
                f"{source} responses",
                1,
                games * 30,
            )
        return set(selected_indices(salt, games))
    if "count" in data:
        needed = search_result_selection_count(data, source)
        return set(selected_indices(salt, needed))
    if isinstance(rows, list):
        games_values = []
        for row_index, row in enumerate(rows):
            games = exact_int(
                row.get("games"),
                f"{source} row {row_index} games",
                1,
                len(WORDS),
            )
            games_values.append(games)
            if "turns" in row:
                validate_turns(
                    row["turns"],
                    games,
                    f"{source} row {row_index} turns",
                )
        return set(selected_indices(salt, max(games_values)))
    raise RuntimeError(f"Unknown result schema in {source}")


def snapshot_result_artifacts() -> dict:
    paths = sorted(
        [
            path
            for path in RESULTS.iterdir()
            if path.suffix in {".json", ".log"}
        ],
        key=lambda path: path.name,
    )
    inventory = []
    frozen_bytes = {}
    parsed_json = {}
    used: set[int] = set()
    for path in paths:
        data = read_stable_bytes(path)
        frozen_bytes[path.name] = data
        inventory.append(
            {
                "name": path.name,
                "sha256": sha256_bytes(data),
                "bytes": len(data),
            }
        )
        if path.suffix == ".json":
            parsed = strict_json_bytes(data, path.name)
            parsed_json[path.name] = parsed
            used.update(used_indices_from_result(parsed, path.name))
    return {
        "inventory": inventory,
        "frozen_bytes": frozen_bytes,
        "parsed_json": parsed_json,
        "used": used,
    }


def public_test_indices(data: bytes) -> set[int]:
    value = strict_json_bytes(data, str(PUBLIC_TEST))
    if not isinstance(value, list) or not value:
        raise RuntimeError("Public test must be a non-empty list")
    indices = []
    for word in value:
        if not isinstance(word, str):
            raise RuntimeError("Public test contains a non-string word")
        index = INDEX.get(word.casefold())
        if index is None:
            raise RuntimeError("Public test contains a word outside vocabulary")
        indices.append(index)
    if len(set(indices)) != len(indices):
        raise RuntimeError("Public test contains duplicate words")
    return set(indices)


def validate_submission_notebook(
    notebook_bytes: bytes,
    source_bytes: bytes,
) -> dict:
    notebook = strict_json_bytes(notebook_bytes, str(CANDIDATE))
    if not isinstance(notebook, dict):
        raise RuntimeError("Candidate notebook must be a JSON object")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or len(cells) != 1:
        raise RuntimeError("Candidate notebook must contain one cell")
    cell = cells[0]
    if not isinstance(cell, dict) or cell.get("cell_type") != "code":
        raise RuntimeError("Candidate notebook cell must be code")
    if cell.get("outputs") != []:
        raise RuntimeError("Candidate notebook must have no outputs")
    if cell.get("execution_count") is not None:
        raise RuntimeError("Candidate notebook must be unexecuted")
    source_value = cell.get("source")
    if isinstance(source_value, list) and all(
        isinstance(line, str) for line in source_value
    ):
        source = "".join(source_value)
    elif isinstance(source_value, str):
        source = source_value
    else:
        raise RuntimeError("Candidate notebook source is invalid")
    if any(
        line.lstrip().startswith("#")
        for line in source.splitlines()
    ):
        raise RuntimeError("Candidate notebook contains comments")
    if source.encode() != source_bytes:
        raise RuntimeError("Candidate notebook and source file differ")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise RuntimeError("Candidate notebook is not valid Python") from error
    allowed_imports = {"json", "os", "sys", "numpy"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] not in allowed_imports:
                    raise RuntimeError(
                        f"Candidate imports forbidden module {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".", 1)[0]
            if module not in allowed_imports:
                raise RuntimeError(
                    f"Candidate imports forbidden module {node.module}"
                )
    lowered = source.casefold()
    for forbidden in [
        "test_public",
        "http://",
        "https://",
        "huggingface",
        "requests",
        "subprocess",
        "socket",
        "pip install",
    ]:
        if forbidden in lowered:
            raise RuntimeError(
                f"Candidate contains forbidden reference {forbidden}"
            )
    if len(notebook_bytes) >= 1_000_000:
        raise RuntimeError("Candidate notebook exceeds one megabyte")
    return {
        "cells": 1,
        "code_only": True,
        "comments": 0,
        "outputs": 0,
        "bytes": len(notebook_bytes),
    }


def validate_incumbent_zip(
    archive_bytes: bytes,
    incumbent_bytes: bytes,
) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            if archive.namelist() != ["solution.ipynb"]:
                raise RuntimeError(
                    "Incumbent ZIP must contain only solution.ipynb"
                )
            if archive.read("solution.ipynb") != incumbent_bytes:
                raise RuntimeError(
                    "Incumbent ZIP notebook differs from root notebook"
                )
            if archive.testzip() is not None:
                raise RuntimeError("Incumbent ZIP integrity check failed")
    except zipfile.BadZipFile as error:
        raise RuntimeError("Incumbent ZIP is invalid") from error


def validate_resource_results(
    parsed_json: dict,
    result_bytes: dict,
) -> dict:
    runs = []
    for name in RESOURCE_RESULTS:
        data = parsed_json.get(name)
        if not isinstance(data, dict):
            raise RuntimeError(f"Missing resource result {name}")
        solution_seconds = finite_number(
            data.get("solution_seconds"),
            f"{name} solution_seconds",
        )
        total_seconds = finite_number(
            data.get("total_seconds"),
            f"{name} total_seconds",
        )
        vmhwm_kib = exact_int(
            data.get("vmhwm_kib"),
            f"{name} vmhwm_kib",
            1,
        )
        cgroup_peak = exact_int(
            data.get("cgroup_memory_peak_bytes"),
            f"{name} cgroup_memory_peak_bytes",
            1,
        )
        swap_peak = exact_int(
            data.get("cgroup_swap_peak_bytes"),
            f"{name} cgroup_swap_peak_bytes",
        )
        swap_current = exact_int(
            data.get("cgroup_swap_current_max_bytes"),
            f"{name} cgroup_swap_current_max_bytes",
        )
        cgroup_samples = exact_int(
            data.get("cgroup_samples"),
            f"{name} cgroup_samples",
            1,
        )
        events = data.get("cgroup_memory_events")
        if not isinstance(events, dict) or not events:
            raise RuntimeError(f"{name} has invalid cgroup events")
        for key, value in events.items():
            exact_int(value, f"{name} cgroup event {key}")
        hard_limit_checks = data.get("hard_limit_checks")
        hard_limit_checks_valid = (
            isinstance(hard_limit_checks, dict)
            and set(hard_limit_checks) == EXPECTED_HARD_LIMIT_CHECKS
            and all(value is True for value in hard_limit_checks.values())
        )
        cgroup = data.get("cgroup")
        properties = (
            cgroup.get("properties")
            if isinstance(cgroup, dict)
            else None
        )
        control_group = (
            cgroup.get("control_group")
            if isinstance(cgroup, dict)
            else None
        )
        cgroup_properties_valid = (
            isinstance(properties, dict)
            and all(
                properties.get(key) == value
                for key, value in EXPECTED_CGROUP_PROPERTIES.items()
            )
            and isinstance(properties.get("MainPID"), str)
            and properties["MainPID"].isdigit()
            and int(properties["MainPID"]) > 0
            and isinstance(control_group, str)
            and control_group.startswith("/")
            and properties.get("ControlGroup") == control_group
        )
        checks = {
            "validator_hash": (
                data.get("validator_sha256") == EXPECTED_VALIDATOR
            ),
            "candidate_hash": (
                data.get("notebook_sha256") == EXPECTED_CANDIDATE
            ),
            "converted_hash": (
                data.get("converted_sha256")
                == EXPECTED_CONVERTED_CANDIDATE
            ),
            "salt": (
                data.get("salt")
                == "consistency-resource-current-patch-v1"
            ),
            "games": (
                data.get("games")
                == THRESHOLDS["resource_games_required"]
            ),
            "responses": (
                data.get("responses")
                == THRESHOLDS["resource_responses_required"]
            ),
            "forced": data.get("force_max_turns") is True,
            "hard_limits": data.get("hard_limits") is True,
            "oracle": data.get("oracle") == "raw",
            "solution_seconds": (
                0.0
                < solution_seconds
                <= THRESHOLDS["resource_solution_seconds_max"]
            ),
            "total_seconds": (
                0.0
                < total_seconds
                <= THRESHOLDS["resource_total_seconds_max"]
            ),
            "vmhwm_kib": (
                0
                < vmhwm_kib
                <= THRESHOLDS["resource_vmhwm_kib_max"]
            ),
            "cgroup_memory_peak": (
                0
                < cgroup_peak
                <= THRESHOLDS[
                    "resource_cgroup_memory_peak_bytes_max"
                ]
            ),
            "cgroup_samples": cgroup_samples > 0,
            "swap": swap_peak == 0 and swap_current == 0,
            "memory_events": all(value == 0 for value in events.values()),
            "hard_limit_checks": hard_limit_checks_valid,
            "cgroup_properties": cgroup_properties_valid,
            "stderr": data.get("stderr") == "",
        }
        runs.append(
            {
                "name": name,
                "sha256": sha256_bytes(result_bytes[name]),
                "solution_seconds": solution_seconds,
                "total_seconds": total_seconds,
                "vmhwm_kib": vmhwm_kib,
                "cgroup_memory_peak_bytes": cgroup_peak,
                "cgroup_samples": cgroup_samples,
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    return {
        "runs": runs,
        "pass": (
            len(runs) == THRESHOLDS["resource_runs_required"]
            and all(run["pass"] for run in runs)
        ),
    }


def configuration_commitment() -> str:
    return commitment(
        {
            "cell_names": CELL_NAMES,
            "cell_size": CELL_SIZE,
            "thresholds": THRESHOLDS,
            "namespace_commitments": {
                "gate": sha256_bytes(GATE_SALT.encode()),
                "mask": sha256_bytes(MASK_SALT.encode()),
                "spectral": sha256_bytes(SPECTRAL_SALT.encode()),
                "bootstrap": sha256_bytes(BOOTSTRAP_SALT.encode()),
            },
            "manual_used_salts_commitment": commitment(
                MANUAL_USED_SALTS
            ),
            "resource_result_names": RESOURCE_RESULTS,
            "expected_validator_sha256": EXPECTED_VALIDATOR,
            "expected_converted_candidate_sha256": (
                EXPECTED_CONVERTED_CANDIDATE
            ),
            "expected_hard_limit_checks": sorted(
                EXPECTED_HARD_LIMIT_CHECKS
            ),
            "expected_cgroup_properties": (
                EXPECTED_CGROUP_PROPERTIES
            ),
        }
    )


def build_audit_state() -> dict:
    script_bytes = read_stable_bytes(Path(__file__))
    baseline_bytes = read_stable_bytes(BASELINE)
    incumbent_bytes = read_stable_bytes(ROOT_INCUMBENT)
    incumbent_zip_bytes = read_stable_bytes(ROOT_INCUMBENT_ZIP)
    candidate_bytes = read_stable_bytes(CANDIDATE)
    candidate_source_bytes = read_stable_bytes(CANDIDATE_SOURCE)
    validator_bytes = read_stable_bytes(VALIDATOR)
    vocabulary_bytes = read_stable_bytes(VOCABULARY)
    embeddings_bytes = read_stable_bytes(EMBEDDINGS)
    public_test_bytes = read_stable_bytes(PUBLIC_TEST)
    hashes = {
        "gate_script": sha256_bytes(script_bytes),
        "vocabulary": sha256_bytes(vocabulary_bytes),
        "embeddings": sha256_bytes(embeddings_bytes),
        "public_test": sha256_bytes(public_test_bytes),
        "baseline": sha256_bytes(baseline_bytes),
        "root_incumbent": sha256_bytes(incumbent_bytes),
        "root_incumbent_zip": sha256_bytes(incumbent_zip_bytes),
        "candidate": sha256_bytes(candidate_bytes),
        "candidate_source": sha256_bytes(candidate_source_bytes),
        "validator": sha256_bytes(validator_bytes),
    }
    if hashes["baseline"] != EXPECTED_BASELINE:
        raise RuntimeError("Baseline hash changed")
    if hashes["root_incumbent"] != EXPECTED_BASELINE:
        raise RuntimeError("Root incumbent hash changed")
    if hashes["root_incumbent_zip"] != EXPECTED_INCUMBENT_ZIP:
        raise RuntimeError("Root incumbent ZIP hash changed")
    if incumbent_bytes != baseline_bytes:
        raise RuntimeError("Root incumbent differs from baseline snapshot")
    if hashes["candidate"] != EXPECTED_CANDIDATE:
        raise RuntimeError("Candidate hash changed")
    if hashes["candidate_source"] != EXPECTED_CANDIDATE_SOURCE:
        raise RuntimeError("Candidate source hash changed")
    if hashes["validator"] != EXPECTED_VALIDATOR:
        raise RuntimeError("Resource validator hash changed")
    if vocabulary_bytes != VOCABULARY_BYTES:
        raise RuntimeError("Vocabulary changed after module import")
    notebook = validate_submission_notebook(
        candidate_bytes,
        candidate_source_bytes,
    )
    validate_incumbent_zip(incumbent_zip_bytes, incumbent_bytes)
    artifacts = snapshot_result_artifacts()
    used = set(artifacts["used"])
    for salt, count in MANUAL_USED_SALTS:
        used.update(selected_indices(salt, count))
    public_indices = public_test_indices(public_test_bytes)
    used.update(public_indices)
    if len(used) != EXPECTED_USED_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_USED_COUNT} used secrets, found {len(used)}"
        )
    remaining_count = len(WORDS) - len(used)
    if remaining_count != EXPECTED_REMAINING_COUNT:
        raise RuntimeError(
            "Unexpected untouched-secret count: "
            f"{remaining_count}"
        )
    if remaining_count - GATE_SIZE != EXPECTED_RESERVE_COUNT:
        raise RuntimeError("Unexpected reserve size")
    resources = validate_resource_results(
        artifacts["parsed_json"],
        artifacts["frozen_bytes"],
    )
    if not resources["pass"]:
        raise RuntimeError("Resource proof did not pass")
    inventory = artifacts["inventory"]
    commitments = {
        "configuration": configuration_commitment(),
        "result_inventory": commitment(inventory),
        "used_indices": commitment(sorted(used)),
        "manual_used_salts": commitment(MANUAL_USED_SALTS),
        "public_test_indices": commitment(sorted(public_indices)),
    }
    return {
        "script_bytes": script_bytes,
        "baseline_bytes": baseline_bytes,
        "incumbent_bytes": incumbent_bytes,
        "incumbent_zip_bytes": incumbent_zip_bytes,
        "candidate_bytes": candidate_bytes,
        "candidate_source_bytes": candidate_source_bytes,
        "validator_bytes": validator_bytes,
        "vocabulary_bytes": vocabulary_bytes,
        "embeddings_bytes": embeddings_bytes,
        "public_test_bytes": public_test_bytes,
        "result_bytes": artifacts["frozen_bytes"],
        "inventory": inventory,
        "used": used,
        "public_indices": public_indices,
        "hashes": hashes,
        "commitments": commitments,
        "resources": resources,
        "notebook": notebook,
        "remaining_count": remaining_count,
    }


def audit_report(state: dict) -> dict:
    return {
        "ready": True,
        "candidate_sha256": state["hashes"]["candidate"],
        "baseline_sha256": state["hashes"]["baseline"],
        "gate_script_sha256": state["hashes"]["gate_script"],
        "notebook": state["notebook"],
        "provenance": {
            "used_count": len(state["used"]),
            "remaining_count": state["remaining_count"],
            "gate_count": GATE_SIZE,
            "reserve_count": EXPECTED_RESERVE_COUNT,
            "public_test_excluded": True,
            "result_artifact_count": len(state["inventory"]),
            "result_inventory_commitment": (
                state["commitments"]["result_inventory"]
            ),
            "used_indices_commitment": (
                state["commitments"]["used_indices"]
            ),
        },
        "resources": state["resources"],
        "thresholds": THRESHOLDS,
        "configuration_commitment": (
            state["commitments"]["configuration"]
        ),
        "marker": str(MARKER_PATH.relative_to(ROOT)),
        "output": str(OUTPUT_PATH.relative_to(ROOT)),
    }


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def write_atomic_new(path: Path, data: bytes) -> None:
    temporary = path.parent / (
        f".{path.name}.tmp-{os.getpid()}-{time.monotonic_ns()}"
    )
    write_exclusive(temporary, data)
    try:
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        fsync_directory(path.parent)


def claim_gate(state: dict) -> dict:
    FINAL_GATE.mkdir(parents=True, exist_ok=True)
    fsync_directory(FINAL_GATE.parent)
    if OUTPUT_PATH.exists():
        raise RuntimeError("Terminal gate output already exists")
    marker = {
        "state": "opened",
        "opened_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S%z",
            time.localtime(),
        ),
        "gate_script_sha256": state["hashes"]["gate_script"],
        "baseline_sha256": state["hashes"]["baseline"],
        "candidate_sha256": state["hashes"]["candidate"],
        "configuration_commitment": (
            state["commitments"]["configuration"]
        ),
        "result_inventory_commitment": (
            state["commitments"]["result_inventory"]
        ),
        "used_indices_commitment": (
            state["commitments"]["used_indices"]
        ),
        "resource_proofs_passed": state["resources"]["pass"],
        "crash_policy": "terminal_reject_no_retry",
    }
    data = canonical_bytes(marker) + b"\n"
    write_exclusive(MARKER_PATH, data)
    marker["sha256"] = sha256_bytes(data)
    return marker


def build_cells(
    used: set[int],
    marker_sha256: str,
) -> tuple[list[dict], dict]:
    marker_bytes = read_stable_bytes(MARKER_PATH)
    if sha256_bytes(marker_bytes) != marker_sha256:
        raise RuntimeError("Gate marker changed before target derivation")
    marker = strict_json_bytes(marker_bytes, str(MARKER_PATH))
    if (
        not isinstance(marker, dict)
        or marker.get("state") != "opened"
        or marker.get("gate_script_sha256")
        != sha256_bytes(read_stable_bytes(Path(__file__)))
        or marker.get("candidate_sha256") != EXPECTED_CANDIDATE
        or marker.get("baseline_sha256") != EXPECTED_BASELINE
        or marker.get("crash_policy") != "terminal_reject_no_retry"
    ):
        raise RuntimeError("Gate marker is invalid")
    remaining = [
        index
        for index in range(len(WORDS))
        if index not in used
    ]
    if len(remaining) != EXPECTED_REMAINING_COUNT:
        raise RuntimeError("Untouched pool changed before gate selection")
    remaining.sort(
        key=lambda index: hashlib.sha256(
            (GATE_SALT + "\0" + WORDS[index]).encode()
        ).digest()
    )
    gate = remaining[:GATE_SIZE]
    reserve = remaining[GATE_SIZE:]
    if len(gate) != GATE_SIZE or len(reserve) != EXPECTED_RESERVE_COUNT:
        raise RuntimeError("Gate or reserve size is invalid")
    if len(set(gate)) != GATE_SIZE or set(gate) & used:
        raise RuntimeError("Gate targets are not unique and untouched")
    cells = []
    for cell_index, name in enumerate(CELL_NAMES):
        start = cell_index * CELL_SIZE
        indices = gate[start : start + CELL_SIZE]
        blocks = [
            position % 5 if cell_index >= 4 else None
            for position in range(CELL_SIZE)
        ]
        cells.append(
            {
                "index": cell_index,
                "name": name,
                "secret_indices": indices,
                "blocks": blocks,
            }
        )
    committed_cells = [
        {
            "index": cell["index"],
            "name": cell["name"],
            "secret_indices": cell["secret_indices"],
            "blocks": cell["blocks"],
        }
        for cell in cells
    ]
    return cells, {
        "remaining_indices": commitment(remaining),
        "gate_cells": commitment(committed_cells),
        "reserve_indices": commitment(reserve),
    }


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.sqrt(np.einsum("ij,ij->i", values, values))
    if np.any(~np.isfinite(norms)) or np.any(norms == 0.0):
        raise RuntimeError("Invalid transformed oracle vectors")
    return np.asarray(values / norms[:, None], dtype=np.float32)


class Oracles:
    def __init__(
        self,
        embeddings_path: Path,
        fit_indices: list[int],
    ):
        embeddings = np.load(embeddings_path, mmap_mode="r")
        if (
            embeddings.ndim != 2
            or embeddings.shape[0] != len(WORDS)
            or embeddings.shape[1] % 5 != 0
        ):
            raise RuntimeError("Invalid embedding matrix")
        self.full = normalize_rows(embeddings)
        fit = np.asarray(fit_indices, dtype=np.int64)
        mean = self.full[fit].mean(axis=0)
        self.center = normalize_rows(self.full - 0.75 * mean)
        order = sorted(
            range(self.full.shape[1]),
            key=lambda dimension: hashlib.sha256(
                (MASK_SALT + "\0" + str(dimension)).encode()
            ).digest(),
        )
        keep = np.asarray(
            order[: self.full.shape[1] // 2],
            dtype=np.int64,
        )
        self.mask = normalize_rows(self.full[:, keep])
        centered_fit = self.full[fit] - mean
        _, _, right = np.linalg.svd(
            centered_fit,
            full_matrices=False,
        )
        basis = np.asarray(right[:24], dtype=np.float32)
        component_order = sorted(
            range(24),
            key=lambda component: hashlib.sha256(
                (SPECTRAL_SALT + "\0" + str(component)).encode()
            ).digest(),
        )
        signs = np.empty(24, dtype=np.float32)
        signs[component_order[:12]] = 1.0
        signs[component_order[12:]] = -1.0
        centered = self.full - mean
        coefficients = centered @ basis.T
        factors = np.exp(0.25 * signs).astype(np.float32) - 1.0
        self.spectral = normalize_rows(
            mean + centered + (coefficients * factors) @ basis
        )
        self.blocks = []
        self.block_centers = []
        self.block_density = []
        width = embeddings.shape[1] // 5
        for block in range(5):
            values = normalize_rows(
                embeddings[:, block * width : (block + 1) * width]
            )
            self.blocks.append(values)
            block_mean = values[fit].mean(axis=0)
            self.block_centers.append(
                normalize_rows(values - 0.75 * block_mean)
            )
            similarities = np.asarray(
                values @ values.T,
                dtype=np.float32,
            )
            np.fill_diagonal(similarities, -np.inf)
            top = np.partition(
                similarities,
                len(WORDS) - 48,
                axis=1,
            )[:, len(WORDS) - 48 :]
            self.block_density.append(
                np.asarray(top.mean(axis=1), dtype=np.float32)
            )

    def difference(
        self,
        cell: int,
        secret: int,
        first: int,
        second: int,
        block: int | None,
    ) -> float:
        if cell == 0:
            values = self.full
            return float(values[secret] @ values[first]) - float(
                values[secret] @ values[second]
            )
        if cell == 1:
            values = self.center
            return float(values[secret] @ values[first]) - float(
                values[secret] @ values[second]
            )
        if cell == 2:
            values = self.mask
            return float(values[secret] @ values[first]) - float(
                values[secret] @ values[second]
            )
        if cell == 3:
            values = self.spectral
            return float(values[secret] @ values[first]) - float(
                values[secret] @ values[second]
            )
        if block is None or block < 0 or block >= 5:
            raise RuntimeError("Held-block oracle has no valid block")
        if cell == 4:
            values = self.blocks[block]
            return float(values[secret] @ values[first]) - float(
                values[secret] @ values[second]
            )
        if cell == 5:
            held = self.blocks[block]
            difference = 0.60 * (
                float(held[secret] @ held[first])
                - float(held[secret] @ held[second])
            )
            others = 0.0
            for index, values in enumerate(self.blocks):
                if index == block:
                    continue
                others += float(values[secret] @ values[first])
                others -= float(values[secret] @ values[second])
            return difference + 0.10 * others
        if cell == 6:
            values = self.block_centers[block]
            return float(values[secret] @ values[first]) - float(
                values[secret] @ values[second]
            )
        if cell != 7:
            raise RuntimeError("Unknown oracle cell")
        values = self.blocks[block]
        density = self.block_density[block]
        first_score = (
            2.0 * float(values[secret] @ values[first])
            - float(density[first])
        )
        second_score = (
            2.0 * float(values[secret] @ values[second])
            - float(density[second])
        )
        return first_score - second_score


def entries_from_cells(cells: list[dict]) -> list[dict]:
    entries = []
    if len(cells) != len(CELL_NAMES):
        raise RuntimeError("Invalid gate cell count")
    for expected_index, cell in enumerate(cells):
        if (
            cell["index"] != expected_index
            or cell["name"] != CELL_NAMES[expected_index]
            or len(cell["secret_indices"]) != CELL_SIZE
            or len(cell["blocks"]) != CELL_SIZE
        ):
            raise RuntimeError("Invalid gate cell layout")
        for secret, block in zip(
            cell["secret_indices"],
            cell["blocks"],
            strict=True,
        ):
            entries.append(
                {
                    "cell": expected_index,
                    "secret": secret,
                    "block": block,
                }
            )
    if len(entries) != GATE_SIZE:
        raise RuntimeError("Invalid gate entry count")
    return entries


def send(process: subprocess.Popen, message: dict) -> None:
    if process.stdin is None:
        raise RuntimeError("Child stdin unavailable")
    process.stdin.write((json.dumps(message) + "\n").encode())
    process.stdin.flush()


def reject_unsolicited_output(
    process: subprocess.Popen,
    buffer: bytearray,
) -> None:
    if buffer:
        raise RuntimeError("Solution emitted multiple response lines")
    if process.stdout is None:
        raise RuntimeError("Child stdout unavailable")
    ready, _, _ = select.select([process.stdout.fileno()], [], [], 0.0)
    if ready:
        chunk = os.read(process.stdout.fileno(), 1024)
        if chunk:
            raise RuntimeError("Solution emitted unsolicited stdout")
        if process.poll() is not None:
            raise RuntimeError("Solution stopped unexpectedly")


def read_line(
    process: subprocess.Popen,
    buffer: bytearray,
    deadline: float,
) -> str:
    if process.stdout is None:
        raise RuntimeError("Child stdout unavailable")
    descriptor = process.stdout.fileno()
    while True:
        newline = buffer.find(b"\n")
        if newline >= 0:
            line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            try:
                return line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise RuntimeError(
                    "Solution response is not UTF-8"
                ) from error
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Child response timed out")
        ready, _, _ = select.select([descriptor], [], [], remaining)
        if not ready:
            raise TimeoutError("Child response timed out")
        chunk = os.read(descriptor, 1024)
        if not chunk:
            raise RuntimeError("Child stopped before responding")
        buffer.extend(chunk)
        if len(buffer) > 4096:
            raise RuntimeError("Child response exceeded 4096 bytes")


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


def monitor_memory(
    pid: int,
    done: threading.Event,
    result: dict[str, int],
) -> None:
    status_path = Path(f"/proc/{pid}/status")
    peak = 0
    while not done.is_set():
        try:
            for line in status_path.read_text().splitlines():
                if line.startswith("VmHWM:"):
                    peak = max(peak, int(line.split()[1]))
        except (FileNotFoundError, ProcessLookupError):
            break
        done.wait(0.002)
    result["vmhwm_kib"] = peak


def convert_notebook(
    notebook: Path,
    directory: Path,
) -> tuple[Path, float]:
    started = time.monotonic()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "python",
            str(notebook),
            "--output",
            "solution",
            "--output-dir",
            str(directory),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    elapsed = time.monotonic() - started
    path = directory / "solution.py"
    if completed.returncode != 0 or not path.is_file():
        raise RuntimeError(completed.stderr.decode(errors="replace"))
    return path, elapsed


def run_notebook(
    notebook: Path,
    execution_root: Path,
    entries: list[dict],
    oracles: Oracles,
) -> dict:
    workspace = tempfile.TemporaryDirectory(prefix="potato-gate-run-")
    directory = Path(workspace.name)
    script, conversion_seconds = convert_notebook(notebook, directory)
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
    stderr_file = tempfile.TemporaryFile()
    started = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=execution_root,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_file,
        start_new_session=True,
    )
    memory = {}
    done = threading.Event()
    monitor = threading.Thread(
        target=monitor_memory,
        args=(process.pid, done, memory),
        daemon=True,
    )
    monitor.start()
    deadline = started + 120.0
    buffer = bytearray()
    turns = []
    responses = 0
    trailing_stdout = b""
    try:
        for entry in entries:
            secret = entry["secret"]
            reject_unsolicited_output(process, buffer)
            send(process, {"event": "new_game"})
            first = LAMP
            second = POTATO
            found = None
            for turn in range(1, 31):
                reject_unsolicited_output(process, buffer)
                difference = oracles.difference(
                    entry["cell"],
                    secret,
                    first,
                    second,
                    entry["block"],
                )
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
                    raise RuntimeError("Invalid response object")
                proposal = INDEX.get(response["new_word"].casefold())
                if proposal is None:
                    raise RuntimeError("Response word outside vocabulary")
                if proposal == secret:
                    found = turn
                    reject_unsolicited_output(process, buffer)
                    send(process, {"status": "win"})
                    break
                first = winner
                second = proposal
            else:
                reject_unsolicited_output(process, buffer)
                send(process, {"status": "loss"})
            turns.append(found)
        reject_unsolicited_output(process, buffer)
        send(process, {"event": "done"})
    finally:
        stop(process)
        done.set()
        monitor.join(timeout=1)
        if process.stdout is not None:
            trailing_stdout = bytes(buffer) + process.stdout.read()
        stderr_file.seek(0)
        stderr_text = stderr_file.read().decode(errors="replace")
        stderr_file.close()
        workspace.cleanup()
    solution_seconds = time.monotonic() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"Solution exited with code {process.returncode}"
        )
    if trailing_stdout:
        raise RuntimeError("Solution emitted unexpected trailing stdout")
    if len(turns) != len(entries):
        raise RuntimeError("Solution produced an incomplete result")
    return {
        "notebook_sha256": sha256_bytes(notebook.read_bytes()),
        "games": len(entries),
        "responses": responses,
        "turns": turns,
        "conversion_seconds": conversion_seconds,
        "solution_seconds": solution_seconds,
        "total_seconds": conversion_seconds + solution_seconds,
        "vmhwm_kib": memory.get("vmhwm_kib", 0),
        "rusage_children_kib": resource.getrusage(
            resource.RUSAGE_CHILDREN
        ).ru_maxrss,
        "returncode": process.returncode,
        "stderr": stderr_text,
    }


def game_score(turn: int | None) -> float:
    if turn is None:
        return 0.0
    return 1.0 - 0.02 * max(0, turn - 10)


def summarize_turns(turns: list[int | None]) -> dict:
    values = np.asarray(
        [31 if turn is None else turn for turn in turns],
        dtype=np.int64,
    )
    scores = np.asarray(
        [game_score(turn) for turn in turns],
        dtype=np.float64,
    )
    return {
        "games": len(turns),
        "wins": int(np.sum(values <= 30)),
        "win_rate": 100.0 * float(np.mean(values <= 30)),
        "score": 100.0 * float(scores.mean()),
        "p10": 100.0 * float(np.mean(values <= 10)),
        "p20": 100.0 * float(np.mean(values <= 20)),
        "median_turn": float(np.median(values)),
        "p90_turn": float(
            np.quantile(values, 0.9, method="higher")
        ),
    }


def paired_metrics(
    baseline: list[int | None],
    candidate: list[int | None],
) -> tuple[dict, np.ndarray]:
    if len(baseline) != len(candidate) or not baseline:
        raise RuntimeError("Invalid paired result lengths")
    base_values = np.asarray(
        [31 if turn is None else turn for turn in baseline],
        dtype=np.int64,
    )
    candidate_values = np.asarray(
        [31 if turn is None else turn for turn in candidate],
        dtype=np.int64,
    )
    base_scores = np.asarray(
        [game_score(turn) for turn in baseline],
        dtype=np.float64,
    )
    candidate_scores = np.asarray(
        [game_score(turn) for turn in candidate],
        dtype=np.float64,
    )
    deltas = candidate_scores - base_scores
    base_wins = base_values <= 30
    candidate_wins = candidate_values <= 30
    return {
        "score_delta": 100.0 * float(deltas.mean()),
        "win_delta": int(
            np.sum(candidate_wins) - np.sum(base_wins)
        ),
        "faster": int(np.sum(candidate_values < base_values)),
        "identical": int(np.sum(candidate_values == base_values)),
        "slower": int(np.sum(candidate_values > base_values)),
        "rescues": int(np.sum(~base_wins & candidate_wins)),
        "harms": int(np.sum(base_wins & ~candidate_wins)),
        "major_gains": int(
            np.sum(base_values - candidate_values >= 5)
        ),
        "major_harms": int(
            np.sum(candidate_values - base_values >= 5)
        ),
    }, deltas


def bootstrap_lower99(
    deltas_by_stratum: list[np.ndarray],
    replicates: int = 200000,
) -> float:
    if not deltas_by_stratum or any(
        len(deltas) == 0 for deltas in deltas_by_stratum
    ):
        raise RuntimeError("Bootstrap has an empty stratum")
    seed = int.from_bytes(
        hashlib.sha256(BOOTSTRAP_SALT.encode()).digest()[:8],
        "big",
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    values = np.empty(replicates, dtype=np.float64)
    position = 0
    batch_size = 2000
    while position < replicates:
        size = min(batch_size, replicates - position)
        total = np.zeros(size, dtype=np.float64)
        count = 0
        for deltas in deltas_by_stratum:
            choices = generator.integers(
                0,
                len(deltas),
                size=(size, len(deltas)),
            )
            total += deltas[choices].sum(axis=1)
            count += len(deltas)
        values[position : position + size] = total / count
        position += size
    return 100.0 * float(
        np.quantile(values, 0.01, method="lower")
    )


def public_run_metadata(run: dict) -> dict:
    return {
        "notebook_sha256": run["notebook_sha256"],
        "games": run["games"],
        "responses": run["responses"],
        "conversion_seconds": round(run["conversion_seconds"], 4),
        "solution_seconds": round(run["solution_seconds"], 4),
        "total_seconds": round(run["total_seconds"], 4),
        "vmhwm_kib": run["vmhwm_kib"],
        "rusage_children_kib": run["rusage_children_kib"],
        "returncode": run["returncode"],
        "stderr_bytes": len(run["stderr"].encode()),
        "stderr_sha256": sha256_bytes(run["stderr"].encode()),
    }


def analyze(
    cells: list[dict],
    baseline_run: dict,
    candidate_run: dict,
    resources_pass: bool,
    hashes_pass: bool,
    provenance_pass: bool,
) -> dict:
    baseline_turns = baseline_run["turns"]
    candidate_turns = candidate_run["turns"]
    if (
        len(baseline_turns) != GATE_SIZE
        or len(candidate_turns) != GATE_SIZE
    ):
        raise RuntimeError("Gate result length is invalid")
    cell_results = []
    bootstrap_strata = []
    for cell in cells:
        start = cell["index"] * CELL_SIZE
        stop = start + CELL_SIZE
        baseline = baseline_turns[start:stop]
        candidate = candidate_turns[start:stop]
        paired, deltas = paired_metrics(baseline, candidate)
        if cell["index"] < 4:
            bootstrap_strata.append(deltas)
        else:
            blocks = np.asarray(cell["blocks"], dtype=np.int64)
            for block in range(5):
                bootstrap_strata.append(deltas[blocks == block])
        cell_results.append(
            {
                "index": cell["index"],
                "name": cell["name"],
                "baseline": summarize_turns(baseline),
                "candidate": summarize_turns(candidate),
                "paired": paired,
            }
        )
    overall_paired, _ = paired_metrics(
        baseline_turns,
        candidate_turns,
    )
    overall = {
        "baseline": summarize_turns(baseline_turns),
        "candidate": summarize_turns(candidate_turns),
        "paired": overall_paired,
        "bootstrap_lower99_score_delta": bootstrap_lower99(
            bootstrap_strata
        ),
        "bootstrap_strata": len(bootstrap_strata),
    }
    protocol_pass = all(
        run["returncode"] == 0
        and run["games"] == GATE_SIZE
        and 0 < run["responses"] <= GATE_SIZE * 30
        and run["vmhwm_kib"] > 0
        and run["stderr"] == ""
        for run in [baseline_run, candidate_run]
    )
    checks = {
        "overall_score": (
            overall_paired["score_delta"]
            >= THRESHOLDS["overall_score_delta_min"]
        ),
        "bootstrap": (
            overall["bootstrap_lower99_score_delta"] > 0.0
        ),
        "overall_wins": (
            overall_paired["win_delta"]
            >= THRESHOLDS["overall_win_delta_min"]
        ),
        "baseline_discriminating": (
            overall["baseline"]["win_rate"]
            <= THRESHOLDS["baseline_win_rate_max"]
        ),
        "cells": all(
            cell["paired"]["score_delta"]
            >= THRESHOLDS["cell_score_delta_min"]
            and cell["paired"]["win_delta"]
            >= THRESHOLDS["cell_win_delta_min"]
            for cell in cell_results
        ),
        "raw_harms": (
            cell_results[0]["paired"]["harms"]
            <= THRESHOLDS["raw_harms_max"]
        ),
        "resources": resources_pass,
        "hashes": hashes_pass,
        "provenance": provenance_pass,
        "protocol": protocol_pass,
    }
    return {
        "cells": cell_results,
        "overall": overall,
        "checks": checks,
        "promote": all(checks.values()),
    }


def current_hash_checks(state: dict) -> tuple[bool, dict]:
    paths = {
        "gate_script": Path(__file__),
        "vocabulary": VOCABULARY,
        "embeddings": EMBEDDINGS,
        "public_test": PUBLIC_TEST,
        "baseline": BASELINE,
        "root_incumbent": ROOT_INCUMBENT,
        "root_incumbent_zip": ROOT_INCUMBENT_ZIP,
        "candidate": CANDIDATE,
        "candidate_source": CANDIDATE_SOURCE,
        "validator": VALIDATOR,
    }
    checks = {}
    for name, path in paths.items():
        try:
            checks[name] = (
                sha256_bytes(read_stable_bytes(path))
                == state["hashes"][name]
            )
        except (OSError, RuntimeError):
            checks[name] = False
    try:
        current = snapshot_result_artifacts()
        checks["result_inventory"] = (
            current["inventory"] == state["inventory"]
            and all(
                current["frozen_bytes"].get(name) == data
                for name, data in state["result_bytes"].items()
            )
        )
    except (OSError, RuntimeError):
        checks["result_inventory"] = False
    return all(checks.values()), checks


def public_values(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: public_values(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [public_values(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def write_snapshot_workspace(
    state: dict,
    directory: Path,
) -> tuple[Path, Path, Path]:
    dataset = directory / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "vocabulary.json").write_bytes(
        state["vocabulary_bytes"]
    )
    (dataset / "public_embeddings.npy").write_bytes(
        state["embeddings_bytes"]
    )
    baseline = directory / "baseline.ipynb"
    candidate = directory / "candidate.ipynb"
    baseline.write_bytes(state["baseline_bytes"])
    candidate.write_bytes(state["candidate_bytes"])
    return baseline, candidate, dataset / "public_embeddings.npy"


def terminal_failure_result(
    state: dict,
    marker: dict,
    error: Exception,
) -> dict:
    hashes_pass, hash_checks = current_hash_checks(state)
    return {
        "status": "terminal_reject_error",
        "promote": False,
        "completed_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S%z",
            time.localtime(),
        ),
        "marker_sha256": marker["sha256"],
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "hashes": {
            "preopen": state["hashes"],
            "postopen_checks": hash_checks,
            "pass": hashes_pass,
        },
        "commitments": state["commitments"],
        "thresholds": THRESHOLDS,
        "decision": "retain_incumbent",
        "retry_allowed": False,
    }


def open_gate(state: dict) -> dict:
    marker = claim_gate(state)
    try:
        cells, gate_commitments = build_cells(
            state["used"],
            marker["sha256"],
        )
        entries = entries_from_cells(cells)
        with tempfile.TemporaryDirectory(
            prefix="potato-final-gate-"
        ) as temporary:
            execution_root = Path(temporary)
            baseline_path, candidate_path, embeddings_path = (
                write_snapshot_workspace(state, execution_root)
            )
            oracles = Oracles(
                embeddings_path,
                sorted(state["used"]),
            )
            baseline_run = run_notebook(
                baseline_path,
                execution_root,
                entries,
                oracles,
            )
            candidate_run = run_notebook(
                candidate_path,
                execution_root,
                entries,
                oracles,
            )
        snapshot_checks = {
            "baseline": (
                baseline_run["notebook_sha256"]
                == EXPECTED_BASELINE
            ),
            "candidate": (
                candidate_run["notebook_sha256"]
                == EXPECTED_CANDIDATE
            ),
        }
        hashes_pass, hash_checks = current_hash_checks(state)
        provenance_checks = {
            "used_count": len(state["used"]) == EXPECTED_USED_COUNT,
            "remaining_count": (
                state["remaining_count"] == EXPECTED_REMAINING_COUNT
            ),
            "gate_count": len(entries) == GATE_SIZE,
            "reserve_count": (
                state["remaining_count"] - len(entries)
                == EXPECTED_RESERVE_COUNT
            ),
            "public_test_excluded": state["public_indices"] <= state["used"],
            "snapshot_hashes": all(snapshot_checks.values()),
        }
        analysis = analyze(
            cells,
            baseline_run,
            candidate_run,
            state["resources"]["pass"],
            hashes_pass and all(snapshot_checks.values()),
            all(provenance_checks.values()),
        )
        result = {
            "status": "complete",
            "promote": analysis["promote"],
            "decision": (
                "promote_candidate"
                if analysis["promote"]
                else "retain_incumbent"
            ),
            "retry_allowed": False,
            "completed_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z",
                time.localtime(),
            ),
            "marker_sha256": marker["sha256"],
            "hashes": {
                "preopen": state["hashes"],
                "snapshot_checks": snapshot_checks,
                "postopen_checks": hash_checks,
                "pass": hashes_pass and all(snapshot_checks.values()),
            },
            "commitments": {
                **state["commitments"],
                **gate_commitments,
            },
            "configuration": {
                "cell_names": CELL_NAMES,
                "cell_size": CELL_SIZE,
                "gate_size": GATE_SIZE,
                "reserve_size": EXPECTED_RESERVE_COUNT,
                "bootstrap_stratification": "cell_and_held_block",
                "thresholds": THRESHOLDS,
            },
            "provenance": {
                "checks": provenance_checks,
                "used_count": len(state["used"]),
                "remaining_count": state["remaining_count"],
                "result_artifacts": state["inventory"],
                "manual_used_salts_commitment": (
                    state["commitments"]["manual_used_salts"]
                ),
            },
            "resource_proofs": state["resources"],
            "runs": {
                "baseline": public_run_metadata(baseline_run),
                "candidate": public_run_metadata(candidate_run),
            },
            "analysis": analysis,
        }
    except Exception as error:
        result = terminal_failure_result(state, marker, error)
    public_result = public_values(result)
    write_atomic_new(
        OUTPUT_PATH,
        json.dumps(
            public_result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n",
    )
    return public_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["audit", "open"],
    )
    args = parser.parse_args()
    state = build_audit_state()
    if args.command == "audit":
        print(
            json.dumps(
                public_values(audit_report(state)),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = open_gate(state)
    print(
        json.dumps(
            {
                "status": result["status"],
                "promote": result["promote"],
                "decision": result["decision"],
                "retry_allowed": result["retry_allowed"],
                "output": str(OUTPUT_PATH.relative_to(ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
