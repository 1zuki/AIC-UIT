"""
Vietnamese OCR Text Correction

Why this exists:
  A normal autoregressive seq2seq model often learns to hallucinate full legal
  paragraphs from scratch. For OCR correction the safer inductive bias is:

      output = concatenate( predicted_edit_for_each_input_character )

  Most characters should be copied. Only suspicious characters are rewritten,
  deleted, or expanded into a short string. This makes training faster, keeps the
  output close to the input, and usually beats a pure seq2seq model on CER.

defaults:
  - uses only train.csv/test.csv
  - no external data
  - no pretrained LM/model/checkpoint/embeddings
  - single GPU
  - checkpoints/resume
  - outputs submission.csv and submission.zip
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import re
import shutil
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from rapidfuzz.distance import Levenshtein as RFLev
except Exception:
    RFLev = None


PAD_CH = "<pad>"
UNK_CH = "<unk>"
PAD = 0
UNK = 1
IGNORE = -100


# utils
def nfc(x: object) -> str:
    if pd.isna(x):
        return ""
    return unicodedata.normalize("NFC", str(x))


def light_cleanup(s: str) -> str:
    s = nfc(s).replace("\u200b", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+([,.;:!?%\]\)\}])", r"\1", s)
    s = re.sub(r"([\[\(\{])\s+", r"\1", s)
    s = re.sub(r"\s*/\s*", "/", s)
    return s.strip()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_train_csv(path: Path, limit_rows: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"id", "input", "corrected_text"}
    miss = need - set(df.columns)
    
    if miss:
        raise ValueError(f"train.csv missing columns: {sorted(miss)}")
    if limit_rows:
        df = df.head(limit_rows).copy()

    df = df.dropna(subset=["input", "corrected_text"]).copy()
    df["input"] = df["input"].map(nfc)
    df["corrected_text"] = df["corrected_text"].map(nfc)
    df = df[(df["input"].str.len() > 0) & (df["corrected_text"].str.len() > 0)]
    df = df.drop_duplicates(subset=["input", "corrected_text"]).reset_index(drop=True)
    
    return df


def load_test_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"id", "input"}
    miss = need - set(df.columns)

    if miss:
        raise ValueError(f"test.csv missing columns: {sorted(miss)}")

    df["input"] = df["input"].map(nfc)
    return df


def levenshtein(a: str, b: str) -> int:
    if RFLev is not None:
        return RFLev.distance(a, b)
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))

        prev = cur

    return prev[-1]


def cer(preds: Sequence[str], refs: Sequence[str]) -> float:
    edits = 0
    total = 0

    for p, r in zip(preds, refs):
        p = nfc(p)
        r = nfc(r)
        edits += levenshtein(p, r)
        total += max(1, len(r))

    return edits / max(1, total)


# vocav
class CharVocab:
    def __init__(self, stoi: Dict[str, int]):
        self.stoi = stoi
        self.itos = [None] * len(stoi)
        for k, v in stoi.items():
            self.itos[v] = k

    @classmethod
    def build(cls, texts: Iterable[str]) -> "CharVocab":
        chars = set()
        for t in texts:
            chars.update(nfc(t))
        stoi = {PAD_CH: PAD, UNK_CH: UNK}
        for ch in sorted(chars):
            if ch not in stoi:
                stoi[ch] = len(stoi)

        return cls(stoi)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, UNK) for ch in nfc(text)]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"stoi": self.stoi}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharVocab":
        return cls(json.loads(path.read_text(encoding="utf-8"))["stoi"])

    def __len__(self) -> int:
        return len(self.itos)


class EditVocab:
    def __init__(self, seg_to_id: Dict[str, int]):
        self.seg_to_id = seg_to_id
        self.id_to_seg = [None] * len(seg_to_id)
        for k, v in seg_to_id.items():
            self.id_to_seg[v] = k

        self.pad_id = self.seg_to_id[PAD_CH]
        self.empty_id = self.seg_to_id.get("", None)

    @classmethod
    def build(
        cls,
        segment_counter: Counter,
        all_chars: Iterable[str],
        min_count: int = 2,
        max_segment_len: int = 4,
        max_labels: int = 4096,
    ) -> "EditVocab":
        seg_to_id = {PAD_CH: 0}
        # deletion label
        seg_to_id[""] = 1
        # every single target/source char must be representable
        for ch in sorted(set(all_chars)):
            if ch and ch not in seg_to_id:
                seg_to_id[ch] = len(seg_to_id)
        # common multi-character expansions: e.g. "ng", "h ", "KT"...
        items = []
        for seg, c in segment_counter.items():
            if seg in seg_to_id:
                continue
            if 0 < len(seg) <= max_segment_len and c >= min_count:
                items.append((seg, c))

        items.sort(key=lambda x: (-x[1], len(x[0]), x[0]))
        for seg, c in items:
            if len(seg_to_id) >= max_labels:
                break
            seg_to_id[seg] = len(seg_to_id)

        return cls(seg_to_id)

    def encode_segment(self, src_ch: str, seg: str) -> int:
        if seg in self.seg_to_id:
            return self.seg_to_id[seg]
        if src_ch in self.seg_to_id:
            return self.seg_to_id[src_ch]

        return self.empty_id if self.empty_id is not None else 0

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"seg_to_id": self.seg_to_id}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EditVocab":
        return cls(json.loads(path.read_text(encoding="utf-8"))["seg_to_id"])

    def __len__(self) -> int:
        return len(self.id_to_seg)


def align_to_segments(src: str, tgt: str) -> List[str]:
    """
    return one output segment per source character

    concat the returned list approximates tgt, insertions are attached to
    the previous source char when possible, replacements of unequal length are
    distributed proportionally across the source block
    """
    src = nfc(src)
    tgt = nfc(tgt)
    if not src:
        return []

    segs = [""] * len(src)
    sm = SequenceMatcher(a=src, b=tgt, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                segs[i] = src[i]
        elif tag == "delete":
            for i in range(i1, i2):
                segs[i] = ""
        elif tag == "insert":
            ins = tgt[j1:j2]
            if i1 > 0:
                segs[i1 - 1] += ins
            elif len(segs) > 0:
                segs[0] = ins + segs[0]
        elif tag == "replace":
            a_len = i2 - i1
            b = tgt[j1:j2]

            if a_len <= 0:
                continue
            if a_len == len(b):
                for k in range(a_len):
                    segs[i1 + k] = b[k]
            else:
                for k in range(a_len):
                    s0 = round(k * len(b) / a_len)
                    s1 = round((k + 1) * len(b) / a_len)
                    segs[i1 + k] = b[s0:s1]
    return segs # SEGS


def split_text_soft(text: str, max_chars: int) -> List[str]:
    text = nfc(text)
    if len(text) <= max_chars:
        return [text]

    chunks = []
    i = 0
    while i < len(text):
        end = min(len(text), i + max_chars)
        if end == len(text):
            chunks.append(text[i:end])
            break

        window = text[i:end]
        cut = -1

        for pat in [r"[\.\!\?;:]\s+", r"[,\)]\s+", r"\s+"]:
            ms = list(re.finditer(pat, window))
            if ms:
                cut = ms[-1].end()
                break

        if cut < max(64, max_chars // 3):
            cut = max_chars

        chunks.append(text[i:i + cut])
        i += cut

    return [c for c in chunks if c]


def proportional_pair_chunks(src: str, tgt: str, max_chars: int) -> List[Tuple[str, str]]:
    src = nfc(src)
    tgt = nfc(tgt)

    if max(len(src), len(tgt)) <= max_chars:
        return [(src, tgt)]

    src_chunks = split_text_soft(src, max_chars)
    out = []
    src_pos = 0

    for sc in src_chunks:
        s0 = src_pos
        s1 = src_pos + len(sc)
        src_pos = s1
        t0 = round(s0 / max(1, len(src)) * len(tgt))
        t1 = round(s1 / max(1, len(src)) * len(tgt))
        # expand to nearby spaces to reduce cutting words in target
        t0 = max(0, min(len(tgt), t0))
        t1 = max(t0, min(len(tgt), t1))
        out.append((sc, tgt[t0:t1]))

    return out


# optional train only exact memory and word lexicon

_TOKEN_RE = re.compile(r"\w+|[^\w\s]+|\s+", re.UNICODE)
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize_keep_space(s: str) -> List[str]:
    return _TOKEN_RE.findall(nfc(s))


def is_word(tok: str) -> bool:
    return bool(_WORD_RE.fullmatch(tok))


def build_exact_map(df: pd.DataFrame, min_conf: float = 0.999) -> Dict[str, str]:
    counts: Dict[str, Counter] = defaultdict(Counter)
    for src, tgt in df[["input", "corrected_text"]].itertuples(index=False):
        counts[nfc(src)][nfc(tgt)] += 1

    exact = {}
    for src, ctr in counts.items():
        tgt, c = ctr.most_common(1)[0]
        conf = c / sum(ctr.values())
        if conf >= min_conf:
            exact[src] = tgt

    return exact


def build_word_lexicon(
    df: pd.DataFrame,
    min_count: int = 3,
    min_conf: float = 0.78,
    max_token_len: int = 40,
    max_tokens_per_row: int = 420,
) -> Dict[str, str]:
    """ conservative token replacement table learned from train only """
    counts: Dict[str, Counter] = defaultdict(Counter)
    total: Counter = Counter()

    for src, tgt in tqdm(df[["input", "corrected_text"]].itertuples(index=False), total=len(df), desc="Build word lexicon"):
        a = [t for t in tokenize_keep_space(src) if not t.isspace()]
        b = [t for t in tokenize_keep_space(tgt) if not t.isspace()]

        if len(a) > max_tokens_per_row or len(b) > max_tokens_per_row:
            continue

        sm = SequenceMatcher(a=a, b=b, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for tok in a[i1:i2]:
                    if is_word(tok) and len(tok) <= max_token_len:
                        total[tok] += 1
                        counts[tok][tok] += 1
                continue

            aa = a[i1:i2]
            bb = b[j1:j2]
            if len(aa) == len(bb):
                for x, y in zip(aa, bb):
                    if is_word(x) and is_word(y) and len(x) <= max_token_len and len(y) <= max_token_len:
                        total[x] += 1
                        counts[x][y] += 1
    lex = {}

    for noisy, ctr in counts.items():
        clean, c = ctr.most_common(1)[0]
        conf = c / max(1, total[noisy])
        if clean != noisy and c >= min_count and conf >= min_conf:
            lex[noisy] = clean

    return lex


def apply_word_lexicon(text: str, lex: Dict[str, str]) -> str:
    if not lex:
        return text
    out = []
    for tok in tokenize_keep_space(text):
        if is_word(tok):
            out.append(lex.get(tok, tok))
        else:
            out.append(tok)

    return "".join(out)


# Dataset building/cache
@dataclass
class TrainConfig:
    train_csv: str
    test_csv: str = "test.csv"
    output_dir: str = "runs/ocr_transducer"
    seed: int = 42
    val_ratio: float = 0.03
    limit_rows: Optional[int] = None

    max_chars: int = 768
    hard_align_chars: int = 1200
    min_seg_count: int = 2
    max_segment_len: int = 4
    max_labels: int = 4096
    identity_augment_ratio: float = 0.12

    d_model: int = 384
    nhead: int = 8
    layers: int = 6
    ffn_dim: int = 1536
    dropout: float = 0.12

    batch_size: int = 24
    accum_steps: int = 1
    epochs: int = 30
    lr: float = 5e-4
    min_lr_ratio: float = 0.05
    warmup_ratio: float = 0.06
    weight_decay: float = 0.03
    grad_clip: float = 1.0
    amp: bool = True
    num_workers: int = 2

    rewrite_weight: float = 3.5
    delete_insert_weight: float = 4.0
    copy_weight: float = 1.0
    label_smoothing: float = 0.02

    eval_samples: int = 1000
    eval_thresholds: str = "0.25,0.35,0.45,0.55,0.65,0.75"
    keep_epoch_ckpts: int = 5
    resume: Optional[str] = None

    use_word_lexicon: bool = True
    lex_min_count: int = 3
    lex_min_conf: float = 0.78


class TransducerDataset(Dataset):
    def __init__(self, examples: List[Tuple[str, List[int], List[int]]], char_vocab: CharVocab, max_chars: int, training: bool):
        self.examples = examples
        self.char_vocab = char_vocab
        self.max_chars = max_chars
        self.training = training
        self.lengths = [min(len(x[0]), max_chars) for x in examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        src, labels, weights = self.examples[idx]
        if len(src) > self.max_chars:
            if self.training:
                start = random.randint(0, len(src) - self.max_chars)
            else:
                start = 0
            src = src[start:start + self.max_chars]
            labels = labels[start:start + self.max_chars]
            weights = weights[start:start + self.max_chars]

        x = self.char_vocab.encode(src)
        return {"x": x, "y": labels, "w": weights}


def collate_batch(batch):
    B = len(batch)
    L = max(len(b["x"]) for b in batch)
    x = torch.full((B, L), PAD, dtype=torch.long)
    y = torch.full((B, L), IGNORE, dtype=torch.long)
    w = torch.zeros((B, L), dtype=torch.float32)
    mask = torch.zeros((B, L), dtype=torch.bool)

    for i, b in enumerate(batch):
        n = len(b["x"])
        x[i, :n] = torch.tensor(b["x"], dtype=torch.long)
        y[i, :n] = torch.tensor(b["y"], dtype=torch.long)
        w[i, :n] = torch.tensor(b["w"], dtype=torch.float32)
        mask[i, :n] = True

    return {"x": x, "y": y, "w": w, "mask": mask}


def train_val_split(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(df) * val_ratio))) if val_ratio > 0 else 0

    return df.iloc[idx[n_val:]].reset_index(drop=True), df.iloc[idx[:n_val]].reset_index(drop=True)


def build_segment_counter(df: pd.DataFrame, cfg: TrainConfig) -> Counter:
    ctr = Counter()
    for src, tgt in tqdm(df[["input", "corrected_text"]].itertuples(index=False), total=len(df), desc="Scan edit segments"):
        for sc, tc in proportional_pair_chunks(src, tgt, cfg.hard_align_chars):
            segs = align_to_segments(sc, tc)
            for seg in segs:
                ctr[seg] += 1

    return ctr


def build_examples(
    df: pd.DataFrame,
    edit_vocab: EditVocab,
    cfg: TrainConfig,
    training: bool = True,
) -> List[Tuple[str, List[int], List[int]]]:
    examples: List[Tuple[str, List[int], List[int]]] = []
    rng = random.Random(cfg.seed + (1 if training else 999))

    for src, tgt in tqdm(df[["input", "corrected_text"]].itertuples(index=False), total=len(df), desc="Build aligned examples"):
        for sc, tc in proportional_pair_chunks(src, tgt, cfg.hard_align_chars):
            if not sc:
                continue
            segs = align_to_segments(sc, tc)
            y = []
            w = []

            for ch, seg in zip(sc, segs):
                yid = edit_vocab.encode_segment(ch, seg)
                y.append(yid)
                # heavier loss on actual edits, insertions are represented by multi-char segments
                if seg == ch:
                    w.append(int(round(cfg.copy_weight * 100)))
                elif seg == "" or len(seg) != 1:
                    w.append(int(round(cfg.delete_insert_weight * 100)))
                else:
                    w.append(int(round(cfg.rewrite_weight * 100)))

            # split long aligned row into trainable chunks
            if len(sc) <= cfg.max_chars:
                examples.append((sc, y, w))
            else:
                pos = 0
                while pos < len(sc):
                    end = min(len(sc), pos + cfg.max_chars)
                    examples.append((sc[pos:end], y[pos:end], w[pos:end]))
                    pos = end

            # identity augmentation: teaches the model not to over-correct clean-looking text
            if training and cfg.identity_augment_ratio > 0 and rng.random() < cfg.identity_augment_ratio:
                clean = tc if tc else sc

                for ic in split_text_soft(clean, cfg.max_chars):
                    iy = [edit_vocab.encode_segment(ch, ch) for ch in ic]
                    iw = [int(round(cfg.copy_weight * 100))] * len(ic)
                    if ic:
                        examples.append((ic, iy, iw))
    return examples


def prepare_cache(df_train: pd.DataFrame, df_val: pd.DataFrame, cfg: TrainConfig, out_dir: Path):
    char_path = out_dir / "char_vocab.json"
    edit_path = out_dir / "edit_vocab.json"
    train_pkl = out_dir / "train_examples.pkl"
    val_pkl = out_dir / "val_examples.pkl"

    if all(p.exists() for p in [char_path, edit_path, train_pkl, val_pkl]) and not getattr(cfg, "rebuild_cache", False):
        print("[*] Loading cached vocab/examples")
        char_vocab = CharVocab.load(char_path)
        edit_vocab = EditVocab.load(edit_path)
        train_examples = pickle.loads(train_pkl.read_bytes())
        val_examples = pickle.loads(val_pkl.read_bytes())

        return char_vocab, edit_vocab, train_examples, val_examples

    print("[*] Building char vocab")
    char_vocab = CharVocab.build(list(df_train["input"]) + list(df_train["corrected_text"]))
    char_vocab.save(char_path)
    print(f"[*] Char vocab size: {len(char_vocab)}")

    print("[*] Building edit vocab")
    seg_counter = build_segment_counter(df_train, cfg)
    all_chars = set()

    for t in list(df_train["input"]) + list(df_train["corrected_text"]):
        all_chars.update(t)

    edit_vocab = EditVocab.build(
        seg_counter,
        all_chars,
        min_count=cfg.min_seg_count,
        max_segment_len=cfg.max_segment_len,
        max_labels=cfg.max_labels,
    )
    edit_vocab.save(edit_path)
    print(f"[*] Edit labels: {len(edit_vocab)}")
    print("[*] Most common edit segments:", seg_counter.most_common(20))

    train_examples = build_examples(df_train, edit_vocab, cfg, training=True)
    val_examples = build_examples(df_val, edit_vocab, cfg, training=False)
    train_pkl.write_bytes(pickle.dumps(train_examples, protocol=pickle.HIGHEST_PROTOCOL))
    val_pkl.write_bytes(pickle.dumps(val_examples, protocol=pickle.HIGHEST_PROTOCOL))
    print(f"[*] Cached examples: train={len(train_examples):,}, val={len(val_examples):,}")

    return char_vocab, edit_vocab, train_examples, val_examples


# model
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)].to(dtype=x.dtype))


class CharEditTagger(nn.Module):
    def __init__(self, char_vocab_size: int, num_labels: int, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(char_vocab_size, cfg.d_model, padding_idx=PAD)
        self.pos = SinusoidalPE(cfg.d_model, max_len=cfg.max_chars + 128, dropout=cfg.dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, num_labels)
        self.scale = math.sqrt(cfg.d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pad_mask = x.eq(PAD)
        h = self.pos(self.emb(x) * self.scale)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.norm(h)

        return self.head(h)


"""
decoding
eval
checkpoints
"""
def weighted_token_loss(logits, y, weights, label_smoothing: float):
    B, L, C = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * L, C),
        y.reshape(B * L),
        ignore_index=IGNORE,
        reduction="none",
        label_smoothing=label_smoothing,
    ).view(B, L)

    valid = y.ne(IGNORE)
    weights = weights.float() / 100.0
    loss = loss * weights * valid.float()

    return loss.sum() / torch.clamp((weights * valid.float()).sum(), min=1.0)


@torch.no_grad()
def predict_chunk(
    model: CharEditTagger,
    text: str,
    char_vocab: CharVocab,
    edit_vocab: EditVocab,
    device: torch.device,
    threshold: float = 0.45,
    copy_margin: float = 0.08,
) -> str:
    if not text:
        return ""

    x_ids = char_vocab.encode(text)
    x = torch.tensor(x_ids, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(x)[0]
    probs = F.softmax(logits, dim=-1)
    best_prob, best_id = probs.max(dim=-1)
    out = []

    for i, ch in enumerate(text):
        pred_seg = edit_vocab.id_to_seg[int(best_id[i])]
        bp = float(best_prob[i])
        copy_id = edit_vocab.seg_to_id.get(ch)
        copy_p = float(probs[i, copy_id]) if copy_id is not None else 0.0
        # conservative gate: if the model is not clearly confident, copy original char
        if pred_seg != ch:
            if bp < threshold:
                pred_seg = ch
            elif copy_id is not None and copy_p + copy_margin >= bp:
                pred_seg = ch
        
        if pred_seg == PAD_CH:
            pred_seg = ch
        out.append(pred_seg)

    return "".join(out)


@torch.no_grad()
def correct_text(
    model: CharEditTagger,
    text: str,
    char_vocab: CharVocab,
    edit_vocab: EditVocab,
    cfg: TrainConfig,
    device: torch.device,
    threshold: float,
    copy_margin: float,
    word_lexicon: Optional[Dict[str, str]] = None,
) -> str:
    chunks = split_text_soft(text, cfg.max_chars)
    pred = "".join(
        predict_chunk(model, c, char_vocab, edit_vocab, device, threshold=threshold, copy_margin=copy_margin)
        for c in chunks
    )
    pred = light_cleanup(pred)
    if word_lexicon:
        pred = light_cleanup(apply_word_lexicon(pred, word_lexicon))

    return pred


@torch.no_grad()
def evaluate(
    model: CharEditTagger,
    df_val: pd.DataFrame,
    char_vocab: CharVocab,
    edit_vocab: EditVocab,
    cfg: TrainConfig,
    device: torch.device,
    max_samples: int,
    thresholds: Sequence[float],
    copy_margin: float,
    word_lexicon: Optional[Dict[str, str]],
):
    if len(df_val) == 0:
        return 999.0, None, []

    sample = df_val.head(max_samples) if max_samples and len(df_val) > max_samples else df_val
    refs = sample["corrected_text"].tolist()
    inputs = sample["input"].tolist()
    best = (999.0, None, [])

    for th in thresholds:
        preds = [
            correct_text(model, x, char_vocab, edit_vocab, cfg, device, threshold=th, copy_margin=copy_margin, word_lexicon=word_lexicon)
            for x in tqdm(inputs, desc=f"Eval th={th:.2f}")
        ]
        c = cer(preds, refs)
        print(f"[*] Val CER threshold={th:.2f}: {c:.6f} | leaderboard approx={1-c:.6f}")

        if c < best[0]:
            best = (c, th, list(zip(inputs[:3], preds[:3], refs[:3])))

    return best


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int, best_cer: float, best_threshold: float, cfg: TrainConfig):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_cer": best_cer,
        "best_threshold": best_threshold,
        "cfg": asdict(cfg),
    }, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=True)

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_cer", 999.0)), float(ckpt.get("best_threshold", 0.45))


def prune_epoch_ckpts(out_dir: Path, keep: int):
    ckpts = sorted(out_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    while len(ckpts) > keep:
        ckpts.pop(0).unlink(missing_ok=True)


def make_scheduler(optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float):
    def f(step):
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


# modes
def inspect_main(args):
    train = load_train_csv(Path(args.train_csv), args.limit_rows)
    test = load_test_csv(Path(args.test_csv))
    print("train:", train.shape, "test:", test.shape)
    print("train input len:\n", train["input"].str.len().describe())
    print("train target len:\n", train["corrected_text"].str.len().describe())
    print("test input len:\n", test["input"].str.len().describe())
    print("copy train CER sample:")
    sample = train.sample(min(len(train), args.sample), random_state=42)
    c = cer(sample["input"].tolist(), sample["corrected_text"].tolist())
    print(f"  copy CER={c:.6f}, score approx={1-c:.6f}")


def train_main(args):
    cfg = TrainConfig(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        seed=args.seed,
        val_ratio=args.val_ratio,
        limit_rows=args.limit_rows,
        max_chars=args.max_chars,
        hard_align_chars=args.hard_align_chars,
        min_seg_count=args.min_seg_count,
        max_segment_len=args.max_segment_len,
        max_labels=args.max_labels,
        identity_augment_ratio=args.identity_augment_ratio,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        epochs=args.epochs,
        lr=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        amp=not args.no_amp,
        num_workers=args.num_workers,
        rewrite_weight=args.rewrite_weight,
        delete_insert_weight=args.delete_insert_weight,
        copy_weight=args.copy_weight,
        label_smoothing=args.label_smoothing,
        eval_samples=args.eval_samples,
        eval_thresholds=args.eval_thresholds,
        keep_epoch_ckpts=args.keep_epoch_ckpts,
        resume=args.resume,
        use_word_lexicon=not args.no_word_lexicon,
        lex_min_count=args.lex_min_count,
        lex_min_conf=args.lex_min_conf,
    )
    cfg.rebuild_cache = args.rebuild_cache  # type: ignore[attr-defined]

    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    print("[*] Loading data")
    df = load_train_csv(Path(cfg.train_csv), cfg.limit_rows)
    df_train, df_val = train_val_split(df, cfg.val_ratio, cfg.seed)
    print(f"[*] rows: train={len(df_train):,}, val={len(df_val):,}")

    exact_path = out_dir / "exact_map.json"
    exact = build_exact_map(df_train)
    exact_path.write_text(json.dumps(exact, ensure_ascii=False), encoding="utf-8")
    print(f"[*] exact input map: {len(exact):,}")

    word_lex = {}
    if cfg.use_word_lexicon:
        lex_path = out_dir / "word_lexicon.json"
        if lex_path.exists() and not args.rebuild_cache:
            word_lex = json.loads(lex_path.read_text(encoding="utf-8"))
        else:
            word_lex = build_word_lexicon(df_train, min_count=cfg.lex_min_count, min_conf=cfg.lex_min_conf)
            lex_path.write_text(json.dumps(word_lex, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[*] word lexicon entries: {len(word_lex):,}")

    char_vocab, edit_vocab, train_examples, val_examples = prepare_cache(df_train, df_val, cfg, out_dir)
    train_ds = TransducerDataset(train_examples, char_vocab, cfg.max_chars, training=True)
    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("[*] device:", device)

    model = CharEditTagger(len(char_vocab), len(edit_vocab), cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
    total_steps = math.ceil(len(loader) / cfg.accum_steps) * cfg.epochs
    warmup_steps = max(10, int(total_steps * cfg.warmup_ratio))
    scheduler = make_scheduler(optimizer, total_steps, warmup_steps, cfg.min_lr_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    start_epoch = 1
    best = 999.0
    best_th = 0.45

    if cfg.resume:
        ep, best, best_th = load_checkpoint(Path(cfg.resume), model, optimizer, scheduler, scaler, device=device)
        start_epoch = ep + 1
        print(f"[*] resumed from epoch={ep}, best={best:.6f}, best_th={best_th}")

    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,train_loss,val_cer,best_threshold,lr,elapsed_sec\n", encoding="utf-8")

    thresholds = [float(x) for x in cfg.eval_thresholds.split(",") if x.strip()]
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_steps = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}")

        for it, batch in enumerate(pbar, 1):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            w = batch["w"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                logits = model(x)
                loss = weighted_token_loss(logits, y, w, cfg.label_smoothing) / cfg.accum_steps

            scaler.scale(loss).backward()
            total_loss += float(loss.detach().cpu()) * cfg.accum_steps
            n_steps += 1

            if it % cfg.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            pbar.set_postfix(loss=f"{total_loss/max(1,n_steps):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        val_cer, val_th, examples = evaluate(
            model,
            df_val,
            char_vocab,
            edit_vocab,
            cfg,
            device,
            cfg.eval_samples,
            thresholds,
            copy_margin=args.copy_margin,
            word_lexicon=word_lex if cfg.use_word_lexicon else None,
        )

        if val_th is None:
            val_th = best_th

        print(f"[*] Epoch {epoch} best val CER={val_cer:.6f}, score approx={1-val_cer:.6f}, th={val_th}")
        for i, (src, pred, ref) in enumerate(examples[:2]):
            print(f"--- val example {i} ---")
            print("INP:", src[:450])
            print("PRD:", pred[:450])
            print("REF:", ref[:450])

        elapsed = time.time() - t0
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},{total_loss/max(1,n_steps):.8f},{val_cer:.8f},{val_th},{scheduler.get_last_lr()[0]:.8g},{elapsed:.2f}\n")

        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, min(best, val_cer), val_th, cfg)
        save_checkpoint(out_dir / f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, scaler, epoch, min(best, val_cer), val_th, cfg)
        prune_epoch_ckpts(out_dir, cfg.keep_epoch_ckpts)
        if val_cer < best:
            best = val_cer
            best_th = val_th
            save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best, best_th, cfg)
            print(f"[*] new best saved: CER={best:.6f}, threshold={best_th}")

    print("[*] done")


def load_run(run_dir: Path):
    cfg_data = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    valid = set(TrainConfig.__dataclass_fields__.keys())
    cfg_data = {k: v for k, v in cfg_data.items() if k in valid}
    cfg = TrainConfig(**cfg_data)
    char_vocab = CharVocab.load(run_dir / "char_vocab.json")
    edit_vocab = EditVocab.load(run_dir / "edit_vocab.json")
    exact = {}

    if (run_dir / "exact_map.json").exists():
        exact = json.loads((run_dir / "exact_map.json").read_text(encoding="utf-8"))

    word_lex = {}
    if (run_dir / "word_lexicon.json").exists():
        word_lex = json.loads((run_dir / "word_lexicon.json").read_text(encoding="utf-8"))

    return cfg, char_vocab, edit_vocab, exact, word_lex


def infer_main(args):
    ckpt = Path(args.checkpoint)
    run_dir = Path(args.run_dir) if args.run_dir else ckpt.parent
    cfg, char_vocab, edit_vocab, exact, word_lex = load_run(run_dir)

    if args.max_chars:
        cfg.max_chars = args.max_chars

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = CharEditTagger(len(char_vocab), len(edit_vocab), cfg).to(device)
    _, best_cer, best_th = load_checkpoint(ckpt, model, device=device)
    model.eval()
    threshold = args.threshold if args.threshold is not None else best_th
    print(f"[*] loaded {ckpt} | best_cer={best_cer:.6f} | threshold={threshold}")
    print("[*] device:", device)

    df = load_test_csv(Path(args.test_csv or cfg.test_csv))
    preds = []

    for text in tqdm(df["input"].tolist(), desc="Infer"):
        if not args.no_exact and text in exact:
            pred = exact[text]
        else:
            pred = correct_text(
                model,
                text,
                char_vocab,
                edit_vocab,
                cfg,
                device,
                threshold=threshold,
                copy_margin=args.copy_margin,
                word_lexicon=None if args.no_word_lexicon else word_lex,
            )
        if not pred and str(text).strip():
            pred = light_cleanup(str(text))
        preds.append(pred)

    sub = pd.DataFrame({"id": df["id"], "corrected_text": preds})
    out = Path(args.submission)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    print(f"[*] saved {out} rows={len(sub):,}")
    if args.zip:
        zp = Path(args.zip)
        if zp.exists():
            zp.unlink()
        with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            zf.write(out, arcname="submission.csv")
        print(f"[*] saved zip {zp} size={zp.stat().st_size/1024/1024:.2f} MB")

    print(sub.head().to_string())


def rules_main(args):
    train = load_train_csv(Path(args.train_csv), args.limit_rows)
    test = load_test_csv(Path(args.test_csv))
    exact = build_exact_map(train)
    lex = build_word_lexicon(train, min_count=args.lex_min_count, min_conf=args.lex_min_conf)
    print(f"exact={len(exact):,}, lex={len(lex):,}")
    preds = []

    for x in tqdm(test["input"], desc="Rules infer"):
        if x in exact:
            y = exact[x]
        else:
            y = light_cleanup(apply_word_lexicon(x, lex))
        preds.append(y)

    sub = pd.DataFrame({"id": test["id"], "corrected_text": preds})
    out = Path(args.submission)
    sub.to_csv(out, index=False, encoding="utf-8")

    if args.zip:
        with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            zf.write(out, arcname="submission.csv")
    print("saved", out)


# cli
def build_parser():
    p = argparse.ArgumentParser(description="Vietnamese OCR correction copy-edit transducer")
    sub = p.add_subparsers(dest="mode", required=True)

    ins = sub.add_parser("inspect")
    ins.add_argument("--train-csv", default="NLP/train.csv")
    ins.add_argument("--test-csv", default="NLP/test.csv")
    ins.add_argument("--limit-rows", type=int, default=None)
    ins.add_argument("--sample", type=int, default=1000)
    ins.set_defaults(func=inspect_main)

    t = sub.add_parser("train")
    t.add_argument("--train-csv", default="NLP/train.csv")
    t.add_argument("--test-csv", default="NLP/test.csv")
    t.add_argument("--output-dir", default="runs/ocr_transducer_seed42")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--val-ratio", type=float, default=0.03)
    t.add_argument("--limit-rows", type=int, default=None)
    t.add_argument("--max-chars", type=int, default=768)
    t.add_argument("--hard-align-chars", type=int, default=1200)
    t.add_argument("--min-seg-count", type=int, default=2)
    t.add_argument("--max-segment-len", type=int, default=4)
    t.add_argument("--max-labels", type=int, default=4096)
    t.add_argument("--identity-augment-ratio", type=float, default=0.12)
    t.add_argument("--d-model", type=int, default=384)
    t.add_argument("--nhead", type=int, default=8)
    t.add_argument("--layers", type=int, default=6)
    t.add_argument("--ffn-dim", type=int, default=1536)
    t.add_argument("--dropout", type=float, default=0.12)
    t.add_argument("--batch-size", type=int, default=24)
    t.add_argument("--accum-steps", type=int, default=1)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--min-lr-ratio", type=float, default=0.05)
    t.add_argument("--warmup-ratio", type=float, default=0.06)
    t.add_argument("--weight-decay", type=float, default=0.03)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--rewrite-weight", type=float, default=3.5)
    t.add_argument("--delete-insert-weight", type=float, default=4.0)
    t.add_argument("--copy-weight", type=float, default=1.0)
    t.add_argument("--label-smoothing", type=float, default=0.02)
    t.add_argument("--eval-samples", type=int, default=1000)
    t.add_argument("--eval-thresholds", default="0.35,0.45,0.55,0.65")
    t.add_argument("--copy-margin", type=float, default=0.08)
    t.add_argument("--num-workers", type=int, default=2)
    t.add_argument("--keep-epoch-ckpts", type=int, default=5)
    t.add_argument("--resume", default=None)
    t.add_argument("--no-amp", action="store_true")
    t.add_argument("--cpu", action="store_true")
    t.add_argument("--rebuild-cache", action="store_true")
    t.add_argument("--no-word-lexicon", action="store_true")
    t.add_argument("--lex-min-count", type=int, default=3)
    t.add_argument("--lex-min-conf", type=float, default=0.78)
    t.set_defaults(func=train_main)

    inf = sub.add_parser("infer")
    inf.add_argument("--checkpoint", required=True)
    inf.add_argument("--run-dir", default=None)
    inf.add_argument("--test-csv", default=None)
    inf.add_argument("--submission", default="submission.csv")
    inf.add_argument("--zip", default="submission.zip")
    inf.add_argument("--threshold", type=float, default=None)
    inf.add_argument("--copy-margin", type=float, default=0.08)
    inf.add_argument("--max-chars", type=int, default=None)
    inf.add_argument("--cpu", action="store_true")
    inf.add_argument("--no-word-lexicon", action="store_true")
    inf.add_argument("--no-exact", action="store_true")
    inf.set_defaults(func=infer_main)

    r = sub.add_parser("rules")
    r.add_argument("--train-csv", default="NLP/train.csv")
    r.add_argument("--test-csv", default="NLP/test.csv")
    r.add_argument("--submission", default="submission_rules.csv")
    r.add_argument("--zip", default="submission_rules.zip")
    r.add_argument("--limit-rows", type=int, default=None)
    r.add_argument("--lex-min-count", type=int, default=3)
    r.add_argument("--lex-min-conf", type=float, default=0.78)
    r.set_defaults(func=rules_main)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
