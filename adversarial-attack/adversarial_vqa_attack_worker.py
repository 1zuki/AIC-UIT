"""
Local full-run script for VQA adversarial attack.

This script mirrors the Kaggle notebook pipeline end-to-end:
- local dataset/model discovery
- ViT cache pre-check
- one full model worker per GPU (or CPU fallback)
- low-res + TV-smoothed attack (same objective as notebook)
- final PNG compression + optional size guard + submission.zip
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F


def torch_load(path: Path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def clean_state_dict(sd):
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if isinstance(sd, dict) and all(str(k).startswith("module.") for k in sd.keys()):
        sd = {str(k)[7:]: v for k, v in sd.items()}
    return sd


def encode_question(q: str, vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    tokens = q.lower().replace("?", "").split()
    ids = [vocab.get(w, 1) for w in tokens]
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def load_u8(path: Path) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def u8_to_float(x: torch.Tensor) -> torch.Tensor:
    return x.float() / 255.0


def save_png_max(path: Path, x_u8: torch.Tensor) -> None:
    arr = x_u8.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, format="PNG", optimize=True, compress_level=9)


def psnr_u8(clean_u8: torch.Tensor, adv_u8: torch.Tensor) -> torch.Tensor:
    mse = (clean_u8.float() - adv_u8.float()).pow(2).flatten(1).mean(1)
    return 10.0 * torch.log10((255.0 * 255.0) / (mse + 1e-12))


def project_psnr_l2(x0: torch.Tensor, x: torch.Tensor, psnr: float) -> torch.Tensor:
    max_mse = 10.0 ** (-psnr / 10.0)
    delta = x - x0
    mse = delta.pow(2).flatten(1).mean(1).view(-1, 1, 1, 1)
    scale = torch.sqrt(torch.tensor(max_mse, device=x.device, dtype=x.dtype) / (mse + 1e-12))
    scale = torch.clamp(scale, max=1.0)
    return torch.clamp(x0 + delta * scale, 0.0, 1.0)


def quantize_with_psnr_safety(x0: torch.Tensor, adv: torch.Tensor, min_psnr: float) -> Tuple[torch.Tensor, torch.Tensor]:
    clean_u8 = torch.round(x0 * 255.0).clamp(0, 255).to(torch.uint8)
    scale = torch.ones((x0.size(0), 1, 1, 1), device=x0.device, dtype=x0.dtype)

    for _ in range(45):
        cand = torch.clamp(x0 + (adv - x0) * scale, 0.0, 1.0)
        adv_u8 = torch.round(cand * 255.0).clamp(0, 255).to(torch.uint8)
        p = psnr_u8(clean_u8, adv_u8)
        bad = p < min_psnr
        if not bool(bad.any()):
            return adv_u8, p
        scale[bad.view(-1, 1, 1, 1)] *= 0.97

    cand = torch.clamp(x0 + (adv - x0) * scale, 0.0, 1.0)
    adv_u8 = torch.round(cand * 255.0).clamp(0, 255).to(torch.uint8)
    return adv_u8, psnr_u8(clean_u8, adv_u8)


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


class ModelWrapper:
    def __init__(self, model_dir: Path, device: torch.device):
        self.device = device
        model_dir = Path(model_dir)
        sys.path.insert(0, str(model_dir))
        from vit_model import Config, ViTModel

        self.Config = Config
        self.vocab = torch_load(model_dir / "vocab.pth", map_location="cpu")
        self.ans_vocab = torch_load(model_dir / "ans_vocab.pth", map_location="cpu")

        self.model = ViTModel(
            vocab_size=len(self.vocab),
            num_answer=len(self.ans_vocab),
            d_model=Config.D_MODEL,
            n_heads=Config.N_HEADS,
            n_layers=Config.N_LAYERS,
            ff_dim=Config.FF_DIM,
            dropout=Config.DROPOUT,
            max_q_len=Config.MAX_LEN,
        ).to(device)

        sd = clean_state_dict(torch_load(model_dir / "model.pth", map_location=device))
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(f"[model] missing={len(missing)} unexpected={len(unexpected)}", flush=True)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def logits(self, raw_float: torch.Tensor, q_tokens: torch.Tensor) -> torch.Tensor:
        return self.model(self.preprocess(raw_float), q_tokens)


def grouped_objective(logits: torch.Tensor, y: torch.Tensor, map_idx: torch.Tensor, batch_images: int, tau: float = 0.35) -> torch.Tensor:
    n = logits.size(0)
    ar = torch.arange(n, device=logits.device)

    correct = logits[ar, y]
    masked = logits.clone()
    masked[ar, y] = -1e9
    other = masked.max(1).values
    margin = other - correct

    ce = F.cross_entropy(logits, y, reduction="none")

    softmins = []
    for b in range(batch_images):
        mb = margin[map_idx == b]
        if mb.numel():
            softmins.append(-tau * torch.logsumexp(-mb / tau, dim=0))
    softmin = torch.stack(softmins).mean() if softmins else margin.mean()

    return ce.mean() + 0.70 * margin.mean() + 0.85 * softmin


@torch.no_grad()
def predict(wrapper: ModelWrapper, x_float: torch.Tensor, q_tokens: torch.Tensor, map_idx: torch.Tensor, chunk: int = 48) -> torch.Tensor:
    preds: List[torch.Tensor] = []
    for s in range(0, q_tokens.size(0), chunk):
        q = q_tokens[s : s + chunk]
        mi = map_idx[s : s + chunk]
        logits = wrapper.logits(x_float[mi], q)
        preds.append(logits.argmax(1))
    return torch.cat(preds, dim=0)


@torch.no_grad()
def score_candidate(
    wrapper: ModelWrapper,
    adv: torch.Tensor,
    q_tokens: torch.Tensor,
    map_idx: torch.Tensor,
    clean_y: torch.Tensor,
    batch_images: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = wrapper.logits(adv[map_idx], q_tokens)
    pred = logits.argmax(1)

    n = logits.size(0)
    ar = torch.arange(n, device=logits.device)
    correct = logits[ar, clean_y]
    masked = logits.clone()
    masked[ar, clean_y] = -1e9
    other = masked.max(1).values
    margin = other - correct

    scores = []
    succ_rates = []
    for b in range(batch_images):
        m = map_idx == b
        if m.any():
            sr = (pred[m] != clean_y[m]).float().mean()
            sc = 100.0 * sr + margin[m].mean() + 0.75 * margin[m].min()
            scores.append(sc)
            succ_rates.append(sr)
        else:
            scores.append(torch.tensor(-1e9, device=adv.device))
            succ_rates.append(torch.tensor(0.0, device=adv.device))
    return torch.stack(scores), torch.stack(succ_rates), pred


def make_adv_from_lowres(x0: torch.Tensor, z_low: torch.Tensor, psnr: float) -> Tuple[torch.Tensor, torch.Tensor]:
    delta = F.interpolate(z_low, size=x0.shape[-2:], mode="bicubic", align_corners=False)
    x = torch.clamp(x0 + delta, 0.0, 1.0)
    x = project_psnr_l2(x0, x, psnr)
    delta = x - x0
    return x, delta


def attack_batch(wrapper: ModelWrapper, x0: torch.Tensor, q_tokens: torch.Tensor, map_idx: torch.Tensor, clean_y: torch.Tensor, cfg: Dict[str, float]) -> torch.Tensor:
    device = x0.device
    B = x0.size(0)
    best_adv = x0.clone()
    best_score = torch.full((B,), -1e9, device=device)

    for r in range(cfg["restarts"]):
        if r == 0:
            init = torch.zeros((B, 3, cfg["low_res"], cfg["low_res"]), device=device)
        else:
            init = torch.empty((B, 3, cfg["low_res"], cfg["low_res"]), device=device).uniform_(-1.25 / 255.0, 1.25 / 255.0)

        z = init.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=cfg["lr"], betas=(0.85, 0.999), eps=1e-8)

        for step in range(cfg["steps"]):
            opt.zero_grad(set_to_none=True)
            adv, delta = make_adv_from_lowres(x0, z, cfg["opt_psnr"])

            logits = wrapper.logits(adv[map_idx], q_tokens)
            attack_obj = grouped_objective(logits, clean_y, map_idx, B)

            smooth = tv_loss(delta)
            l2 = delta.pow(2).mean()
            loss = -attack_obj + cfg["tv_lambda"] * smooth + cfg["l2_lambda"] * l2
            loss.backward()

            if z.grad is not None:
                z.grad.data = torch.nan_to_num(z.grad.data, nan=0.0, posinf=0.0, neginf=0.0)

            opt.step()

            with torch.no_grad():
                z.clamp_(-8.0 / 255.0, 8.0 / 255.0)

            if ((step + 1) % cfg["eval_every"] == 0) or (step + 1 == cfg["steps"]):
                with torch.no_grad():
                    cand, _ = make_adv_from_lowres(x0, z, cfg["opt_psnr"])
                    scores, _, _ = score_candidate(wrapper, cand, q_tokens, map_idx, clean_y, B)
                    better = scores > best_score
                    if bool(better.any()):
                        best_adv[better] = cand.detach()[better]
                        best_score[better] = scores[better]

    return best_adv.detach()


def build_groups(data_dir: Path) -> List[Tuple[str, List[dict]]]:
    with open(Path(data_dir) / "questions" / "test.json", "r") as f:
        questions = json.load(f)["questions"]
    groups: "OrderedDict[str, List[dict]]" = OrderedDict()
    for item in questions:
        groups.setdefault(item["image_filename"], []).append(item)
    return list(groups.items())


def run_attack_rank(rank: int, world: int, gpu: int, data_dir: str, model_dir: str, out_root: str, cfg: Dict[str, float]) -> None:
    random.seed(cfg["seed"] + rank)
    np.random.seed(cfg["seed"] + rank)
    torch.manual_seed(cfg["seed"] + rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        device = torch.device(f"cuda:{gpu}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")

    out_dir = Path(out_root) / f"gpu{rank}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rank {rank}] device={device} out={out_dir}", flush=True)

    wrapper = ModelWrapper(Path(model_dir), device)
    max_len = wrapper.Config.MAX_LEN

    all_groups = build_groups(Path(data_dir))
    if cfg.get("limit_images"):
        all_groups = all_groups[: int(cfg["limit_images"])]
    groups = all_groups[rank::world]
    print(f"[rank {rank}] groups={len(groups)} / total={len(all_groups)}", flush=True)

    total_q = 0
    success_q = 0
    psnr_values: List[float] = []
    image_count = 0
    per_image = []

    pbar = tqdm(range(0, len(groups), cfg["batch_images"]), desc=f"rank{rank}/gpu{gpu}", position=rank)
    for start in pbar:
        batch_groups = groups[start : start + cfg["batch_images"]]
        B = len(batch_groups)

        clean_u8_list = []
        q_list = []
        map_list = []
        filename_list = []
        q_count_per_img = []

        for bi, (fn, items) in enumerate(batch_groups):
            filename_list.append(fn)
            clean_u8 = load_u8(Path(data_dir) / "images" / fn)
            clean_u8_list.append(clean_u8)
            q_count_per_img.append(len(items))

            for item in items:
                q_list.append(encode_question(item["question"], wrapper.vocab, max_len))
                map_list.append(bi)

        clean_u8 = torch.stack(clean_u8_list, dim=0).to(device, non_blocking=True)
        x0 = u8_to_float(clean_u8)
        q_tokens = torch.stack(q_list, dim=0).to(device, non_blocking=True)
        map_idx = torch.tensor(map_list, dtype=torch.long, device=device)

        with torch.no_grad():
            clean_y = predict(wrapper, x0, q_tokens, map_idx)

        adv = attack_batch(wrapper, x0, q_tokens, map_idx, clean_y, cfg)
        adv_u8, psnrs = quantize_with_psnr_safety(x0, adv, cfg["save_min_psnr"])
        adv_float_saved = u8_to_float(adv_u8).to(device)

        with torch.no_grad():
            adv_pred = predict(wrapper, adv_float_saved, q_tokens, map_idx)
            success = adv_pred != clean_y

        offset = 0
        for bi, fn in enumerate(filename_list):
            save_png_max(out_dir / fn, adv_u8[bi].detach().cpu())

            k = q_count_per_img[bi]
            succ_i = int(success[offset : offset + k].sum().item())
            total_i = int(k)
            offset += k

            ps = float(psnrs[bi].detach().cpu().item())
            total_q += total_i
            success_q += succ_i
            psnr_values.append(ps)
            image_count += 1
            per_image.append({"file": fn, "success": succ_i, "total": total_i, "psnr": ps})

        pbar.set_postfix(
            {
                "ASR": f"{100.0 * success_q / max(total_q, 1):.1f}%",
                "minPSNR": f"{min(psnr_values):.2f}",
            }
        )

        del clean_u8, x0, q_tokens, map_idx, clean_y, adv, adv_u8, adv_float_saved, adv_pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stats = {
        "rank": rank,
        "gpu": gpu,
        "images": image_count,
        "questions": total_q,
        "success": success_q,
        "asr": success_q / max(total_q, 1),
        "min_psnr": min(psnr_values) if psnr_values else None,
        "mean_psnr": sum(psnr_values) / max(len(psnr_values), 1),
        "per_image": per_image,
    }
    with open(out_dir / f"stats_rank{rank}.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[rank {rank}] done ASR={100.0 * stats['asr']:.2f}% minPSNR={stats['min_psnr']}", flush=True)


def detect_gpu_count_no_torch_cuda() -> int:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis and vis.strip() and vis.strip() != "-1":
        parts = [x for x in vis.split(",") if x.strip()]
        if parts:
            return len(parts)
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
        return len([ln for ln in out.splitlines() if ln.strip().startswith("GPU ")])
    except Exception:
        return 0


def is_dataset_dir(p: Path) -> bool:
    return (p / "images").is_dir() and (p / "questions" / "test.json").exists()


def is_model_dir(p: Path) -> bool:
    needed = ["vit_model.py", "model.pth", "vocab.pth", "ans_vocab.pth"]
    return all((p / n).exists() for n in needed)


def find_named_file(candidates: Iterable[str], roots: Iterable[Path]) -> Optional[Path]:
    want = {c.lower() for c in candidates}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.name.lower() in want:
                return p
    return None


def extract_zip_once(zip_path: Path, dest: Path) -> None:
    marker = dest / f".extracted_{zip_path.stem}"
    if marker.exists():
        print(f"Already extracted: {zip_path.name}")
        return
    print(f"Extracting {zip_path} -> {dest}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    marker.write_text("ok")


def find_dataset_dir(data_hint: Path, extract_dir: Path, work: Path) -> Path:
    roots = [data_hint, extract_dir, work]
    for root in roots:
        if not root.exists():
            continue
        for p in [root] + [x for x in root.rglob("*") if x.is_dir()]:
            if is_dataset_dir(p):
                return p
    raise FileNotFoundError("Could not find dataset dir. Need images/ and questions/test.json.")


def find_model_dir(model_hint: Path, extract_dir: Path, work: Path) -> Path:
    roots = [model_hint, extract_dir, work]
    for root in roots:
        if not root.exists():
            continue
        for p in [root] + [x for x in root.rglob("*") if x.is_dir()]:
            if is_model_dir(p):
                return p
    raise FileNotFoundError("Could not find model dir. Need vit_model.py/model.pth/vocab.pth/ans_vocab.pth.")


def pre_cache_vit_weights(torch_home: Path) -> None:
    print("torch:", torch.__version__)
    print("TORCH_HOME:", torch_home)
    try:
        import torchvision.models as tvm

        print("Pre-caching torchvision ViT_B_16_Weights.DEFAULT...")
        tmp_vit = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.DEFAULT)
        del tmp_vit
        print("ViT weights cached")
    except Exception as e:
        raise RuntimeError(
            "Could not pre-cache ViT weights. If offline, make sure vit_b_16-c867db91.pth exists under "
            f"{torch_home / 'hub' / 'checkpoints'}"
        ) from e


def make_zip_from_pngs(zip_path: Path, png_map: Dict[str, Path], expected_files: List[str]) -> float:
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in expected_files:
            zf.write(png_map[name], arcname=name)
    return zip_path.stat().st_size / (1024 * 1024)


def collect_outputs(out_root: Path) -> Dict[str, Path]:
    pngs: Dict[str, Path] = {}
    for shard in sorted(Path(out_root).glob("gpu*")):
        for p in shard.glob("*.png"):
            pngs[p.name] = p
    return pngs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Notebook-equivalent VQA adversarial attack runner")

    ap.add_argument("--project-root", type=str, default=None)
    ap.add_argument("--work-dir", type=str, default=None)
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--model-dir", type=str, default=None)
    ap.add_argument("--out-root", type=str, default=None)
    ap.add_argument("--submission-zip", type=str, default=None)
    ap.add_argument("--torch-cache", type=str, default=None)

    ap.add_argument("--preset", choices=["FAST", "COOK", "INSANE"], default="COOK")
    ap.add_argument("--batch-images", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--restarts", type=int, default=None)
    ap.add_argument("--low-res", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--tv-lambda", type=float, default=None)

    ap.add_argument("--opt-psnr", type=float, default=45.08)
    ap.add_argument("--save-min-psnr", type=float, default=45.02)
    ap.add_argument("--l2-lambda", type=float, default=0.15)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-zip-mb", type=float, default=49.20)

    ap.add_argument("--max-gpu-workers", type=int, default=2)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cpu", action="store_true", help="Force CPU single-worker run")

    ap.add_argument("--limit-images", type=int, default=None, help="For smoke tests; process only first N unique images")

    return ap.parse_args()


def resolve_preset(preset: str) -> Dict[str, float]:
    if preset == "FAST":
        return {"batch_images": 8, "steps": 45, "restarts": 1, "low_res": 112, "lr": 0.85 / 255.0, "tv_lambda": 3.0}
    if preset == "COOK":
        return {"batch_images": 6, "steps": 75, "restarts": 2, "low_res": 112, "lr": 0.75 / 255.0, "tv_lambda": 4.0}
    return {"batch_images": 5, "steps": 120, "restarts": 3, "low_res": 112, "lr": 0.65 / 255.0, "tv_lambda": 4.0}


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).expanduser().resolve() if args.project_root else script_dir
    work = Path(args.work_dir).expanduser().resolve() if args.work_dir else project_root

    data_dir_hint = Path(args.data_dir).expanduser().resolve() if args.data_dir else (project_root / "dataset")
    model_dir_hint = Path(args.model_dir).expanduser().resolve() if args.model_dir else (project_root / "Model")

    extract_dir = work / "extracted"
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else (work / "adv_outputs_smooth")
    submission_zip = Path(args.submission_zip).expanduser().resolve() if args.submission_zip else (work / "submission.zip")
    torch_cache = Path(args.torch_cache).expanduser().resolve() if args.torch_cache else (work / "torch_cache")

    os.environ["TORCH_HOME"] = str(torch_cache)

    print("PROJECT_ROOT:", project_root)
    print("WORK:", work)
    print("DATASET_DIR_HINT:", data_dir_hint)
    print("MODEL_DIR_HINT:", model_dir_hint)
    print("TORCH_HOME:", os.environ["TORCH_HOME"])

    preset_vals = resolve_preset(args.preset)
    batch_images = args.batch_images if args.batch_images is not None else preset_vals["batch_images"]
    steps = args.steps if args.steps is not None else preset_vals["steps"]
    restarts = args.restarts if args.restarts is not None else preset_vals["restarts"]
    low_res = args.low_res if args.low_res is not None else preset_vals["low_res"]
    lr = args.lr if args.lr is not None else preset_vals["lr"]
    tv_lambda = args.tv_lambda if args.tv_lambda is not None else preset_vals["tv_lambda"]

    print(
        f"Preset={args.preset} batch_images={batch_images} steps={steps} restarts={restarts} "
        f"low_res={low_res} lr={lr:.8f} tv_lambda={tv_lambda}"
    )

    extract_dir.mkdir(parents=True, exist_ok=True)
    torch_cache.mkdir(parents=True, exist_ok=True)

    data_dir = data_dir_hint if is_dataset_dir(data_dir_hint) else None
    model_dir = model_dir_hint if is_model_dir(model_dir_hint) else None

    if data_dir is None or model_dir is None:
        dataset_zip = find_named_file(["dataset.zip"], [work, project_root])
        model_zip = find_named_file(["Model.zip", "model.zip"], [work, project_root])

        print("dataset_zip:", dataset_zip)
        print("model_zip:", model_zip)

        if data_dir is None and dataset_zip is not None:
            extract_zip_once(dataset_zip, extract_dir)
        if model_dir is None and model_zip is not None:
            extract_zip_once(model_zip, extract_dir)

    if data_dir is None:
        data_dir = find_dataset_dir(data_dir_hint, extract_dir, work)
    if model_dir is None:
        model_dir = find_model_dir(model_dir_hint, extract_dir, work)

    print("DATA_DIR:", data_dir)
    print("MODEL_DIR:", model_dir)

    with open(data_dir / "questions" / "test.json", "r") as f:
        q_data = json.load(f)["questions"]
    unique_files = sorted({x["image_filename"] for x in q_data})
    ordered_group_files = [fn for fn, _ in build_groups(data_dir)]
    print("Questions:", len(q_data), "| unique images:", len(unique_files))

    pre_cache_vit_weights(torch_cache)

    cfg = {
        "batch_images": int(batch_images),
        "steps": int(steps),
        "restarts": int(restarts),
        "low_res": int(low_res),
        "lr": float(lr),
        "tv_lambda": float(tv_lambda),
        "l2_lambda": float(args.l2_lambda),
        "eval_every": int(args.eval_every),
        "opt_psnr": float(args.opt_psnr),
        "save_min_psnr": float(args.save_min_psnr),
        "seed": int(args.seed),
        "limit_images": int(args.limit_images) if args.limit_images else None,
    }

    out_root.mkdir(parents=True, exist_ok=True)
    for p in out_root.glob("gpu*"):
        if p.is_dir():
            shutil.rmtree(p)

    gpu_count = 0 if args.cpu else detect_gpu_count_no_torch_cuda()
    if args.workers is not None:
        world = max(1, int(args.workers))
    else:
        world = max(1, min(args.max_gpu_workers, gpu_count if gpu_count > 0 else 1))

    print("Detected GPUs:", gpu_count, "| workers:", world)

    start_time = time.time()

    if world == 1:
        gpu = 0 if gpu_count > 0 else 0
        run_attack_rank(0, 1, gpu, str(data_dir), str(model_dir), str(out_root), cfg)
    else:
        ctx = mp.get_context("fork")
        procs = []

        for rank in range(world):
            gpu = rank if gpu_count > 0 else 0
            p = ctx.Process(
                target=run_attack_rank,
                args=(rank, world, gpu, str(data_dir), str(model_dir), str(out_root), cfg),
            )
            p.start()
            procs.append(p)

        failed = []
        for rank, p in enumerate(procs):
            p.join()
            print(f"rank {rank} exitcode:", p.exitcode)
            if p.exitcode != 0:
                failed.append((rank, p.exitcode))

        if failed:
            raise RuntimeError(f"Some attack workers failed: {failed}")

    print(f"Attack finished in {(time.time() - start_time) / 60:.1f} min")

    expected_files = ordered_group_files[: cfg["limit_images"]] if cfg["limit_images"] else unique_files

    pngs = collect_outputs(out_root)
    missing = sorted(set(expected_files) - set(pngs))
    extra = sorted(set(pngs) - set(expected_files))

    print("Expected images:", len(expected_files))
    print("Generated images:", len(pngs))
    print("Missing:", len(missing), missing[:5])
    print("Extra:", len(extra), extra[:5])
    if missing:
        raise RuntimeError("Missing required adversarial images.")

    size_mb = make_zip_from_pngs(submission_zip, pngs, expected_files)
    print(f"Initial submission size: {size_mb:.2f} MB")

    if size_mb > args.max_zip_mb:
        print(f"Zip is above {args.max_zip_mb:.2f} MB, starting size guard shrink...")
        shrunk_root = work / "adv_outputs_size_guard"
        if shrunk_root.exists():
            shutil.rmtree(shrunk_root)
        shrunk_root.mkdir(parents=True, exist_ok=True)

        best_size = size_mb
        best_factor = 1.0

        for factor in [0.96, 0.93, 0.90, 0.87, 0.84, 0.80, 0.76, 0.72, 0.68]:
            trial_dir = shrunk_root / f"factor_{factor:.2f}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            trial_map = {}

            min_psnr = 999.0
            for name in expected_files:
                clean = np.array(Image.open(data_dir / "images" / name).convert("RGB"), dtype=np.float32)
                adv = np.array(Image.open(pngs[name]).convert("RGB"), dtype=np.float32)

                shrunk = np.rint(clean + (adv - clean) * factor).clip(0, 255).astype(np.uint8)

                mse = np.mean((clean - shrunk.astype(np.float32)) ** 2)
                psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, 1e-12))
                min_psnr = min(min_psnr, float(psnr))

                out = trial_dir / name
                Image.fromarray(shrunk, mode="RGB").save(out, format="PNG", optimize=True, compress_level=9)
                trial_map[name] = out

            trial_zip = work / f"submission_factor_{factor:.2f}.zip"
            trial_size = make_zip_from_pngs(trial_zip, trial_map, expected_files)
            print(f"factor={factor:.2f} size={trial_size:.2f} MB minPSNR={min_psnr:.2f}")

            if trial_size < best_size:
                best_size = trial_size
                best_factor = factor

            if trial_size <= args.max_zip_mb and min_psnr >= args.save_min_psnr:
                shutil.copy2(trial_zip, submission_zip)
                size_mb = trial_size
                print(f"Using factor={factor:.2f}")
                break

        if size_mb > args.max_zip_mb:
            print(
                f"WARNING: best shrink factor={best_factor:.2f} produced {best_size:.2f} MB, still above cap. "
                #
                "Try LOW_RES=56, PRESET='FAST', or lower steps."
            )
        else:
            print(f"Final guarded submission size: {size_mb:.2f} MB")

    stats_files = sorted(Path(out_root).glob("gpu*/stats_rank*.json"))
    total_q = total_success = total_images = 0
    min_psnr = 999.0

    for sf in stats_files:
        st = json.loads(sf.read_text())
        print(sf.name, "ASR:", f"{100 * st['asr']:.2f}%", "images:", st["images"], "minPSNR:", st["min_psnr"])
        total_q += st["questions"]
        total_success += st["success"]
        total_images += st["images"]
        if st["min_psnr"] is not None:
            min_psnr = min(min_psnr, st["min_psnr"])

    final_size = submission_zip.stat().st_size / (1024 * 1024)
    print("=" * 60)
    print("Final submission:", submission_zip)
    print(f"Final size: {final_size:.2f} MB")
    print("Images:", total_images)
    print("Local attack success vs clean predictions:", f"{100 * total_success / max(total_q, 1):.2f}%")
    print("Local min PSNR before any size-guard shrink:", f"{min_psnr:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
