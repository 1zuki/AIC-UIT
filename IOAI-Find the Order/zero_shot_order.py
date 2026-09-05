import json
import logging
import math
import os
import wave
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperForConditionalGeneration, WhisperProcessor
from transformers.utils import logging as transformers_logging

logging.disable(logging.CRITICAL)
transformers_logging.set_verbosity_error()

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "dataset" / "test_private"
PREFIX_PATH = DATA_DIR / "prefix.json"
WHISPER_PATH = BASE_DIR / "models" / "whisper-small"
QWEN_PATH = BASE_DIR / "models" / "qwen2.5-0.5b"
OUTPUT_PATH = BASE_DIR / "submission.json"
SAMPLE_RATE = 16000
WHISPER_BATCH_SIZE = 8
QWEN_BATCH_SIZE = 8
TARGET_TOKEN_LIMIT = 32
CONTEXT_TURNS = 4
BEAM_WIDTH = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_dtype = torch.float16 if device.type == "cuda" else torch.float32
qwen_dtype = (
    torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported()
    else torch.float16
    if device.type == "cuda"
    else torch.float32
)

torch.manual_seed(0)
if device.type == "cuda":
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def numeric_key(value):
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


def chunk_key(path):
    return int(path.stem.rsplit("_", 1)[1])


def read_wave(path):
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError("unsupported audio sample width")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, rate)
    return np.asarray(audio, dtype=np.float32)


whisper_processor = WhisperProcessor.from_pretrained(
    WHISPER_PATH,
    local_files_only=True,
)
whisper_model = WhisperForConditionalGeneration.from_pretrained(
    WHISPER_PATH,
    local_files_only=True,
    torch_dtype=whisper_dtype,
).to(device).eval()


def transcribe(audios):
    texts = []
    for start in range(0, len(audios), WHISPER_BATCH_SIZE):
        batch = audios[start:start + WHISPER_BATCH_SIZE]
        features = whisper_processor(
            batch,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        kwargs = {
            "input_features": features.input_features.to(device=device, dtype=whisper_dtype),
            "attention_mask": features.attention_mask.to(device),
            "language": "en",
            "task": "transcribe",
            "do_sample": False,
            "num_beams": 1,
            "temperature": 0.0,
            "max_new_tokens": 128,
        }
        try:
            with torch.inference_mode():
                tokens = whisper_model.generate(**kwargs)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            half = max(1, len(batch) // 2)
            texts.extend(transcribe(batch[:half]))
            texts.extend(transcribe(batch[half:]))
            continue
        texts.extend(
            whisper_processor.batch_decode(
                tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
    return [str(text).replace("\n", " ").strip() for text in texts]


qwen_tokenizer = AutoTokenizer.from_pretrained(
    QWEN_PATH,
    local_files_only=True,
)
qwen_model = AutoModelForCausalLM.from_pretrained(
    QWEN_PATH,
    local_files_only=True,
    torch_dtype=qwen_dtype,
).to(device).eval()


def encode_piece(text):
    return qwen_tokenizer.encode(str(text), add_special_tokens=False)


def qwen_next_scores(prefixes, candidates):
    sequences = []
    target_ids = []
    target_lengths = []
    for prefix, candidate in zip(prefixes, candidates):
        left = encode_piece(prefix)
        right = encode_piece(" " + str(candidate).replace("\n", " ").strip())
        right = right[:TARGET_TOKEN_LIMIT]
        if not right:
            right = [qwen_tokenizer.eos_token_id]
        sequence = left + right
        sequences.append(sequence)
        target_ids.append(right)
        target_lengths.append(len(right))
    scores = []
    for start in range(0, len(sequences), QWEN_BATCH_SIZE):
        batch_sequences = sequences[start:start + QWEN_BATCH_SIZE]
        batch_targets = target_ids[start:start + QWEN_BATCH_SIZE]
        batch_lengths = target_lengths[start:start + QWEN_BATCH_SIZE]
        max_length = max(len(sequence) for sequence in batch_sequences)
        max_target = max(batch_lengths)
        input_ids = torch.full(
            (len(batch_sequences), max_length),
            qwen_tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, sequence in enumerate(batch_sequences):
            length = len(sequence)
            input_ids[row, max_length - length:] = torch.tensor(
                sequence,
                dtype=torch.long,
                device=device,
            )
            attention_mask[row, max_length - length:] = 1
        with torch.inference_mode():
            logits = qwen_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=max_target + 1,
                use_cache=False,
            ).logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)
        for row, (target, length) in enumerate(zip(batch_targets, batch_lengths)):
            offset = max_target - length
            values = log_probs[row, offset:offset + length]
            labels = torch.tensor(target, dtype=torch.long, device=device)
            score = values.gather(1, labels.unsqueeze(1)).squeeze(1).mean()
            scores.append(float(score.detach().cpu()))
        del input_ids, attention_mask, logits, log_probs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return scores


def clean_text(text):
    return " ".join(str(text).split())


def make_prefix(path, texts):
    shown = path[-CONTEXT_TURNS:]
    first_position = len(path) - len(shown)
    lines = ["Conversation:"]
    for offset, index in enumerate(shown):
        position = first_position + offset
        speaker = "A" if position % 2 == 0 else "B"
        lines.append("Speaker " + speaker + ": " + clean_text(texts[index]))
    next_speaker = "A" if len(path) % 2 == 0 else "B"
    lines.append("Speaker " + next_speaker + ":")
    return "\n".join(lines)


def order_dialogue(texts, prefix):
    count = len(texts)
    fixed = tuple(int(index) for index in prefix)
    if len(fixed) < 2 or len(set(fixed)) != len(fixed) or any(index < 0 or index >= count for index in fixed):
        raise ValueError("invalid prefix")
    beams = [(0.0, fixed)]
    while len(beams[0][1]) < count:
        expansions = []
        for total, path in beams:
            used = set(path)
            prefix_text = make_prefix(path, texts)
            for index in range(count):
                if index not in used:
                    expansions.append((total, path, index, prefix_text))
        values = qwen_next_scores(
            [expansion[3] for expansion in expansions],
            [texts[expansion[2]] for expansion in expansions],
        )
        candidates = [
            (
                total + value,
                path + (index,),
            )
            for (total, path, index, _), value in zip(expansions, values)
        ]
        candidates.sort(key=lambda item: (-item[0], item[1]))
        beams = candidates[:BEAM_WIDTH]
    return list(beams[0][1])


def to_ranks(order):
    ranks = [0] * len(order)
    for position, index in enumerate(order):
        ranks[index] = position
    return ranks


with PREFIX_PATH.open(encoding="utf-8") as handle:
    prefixes = json.load(handle)

answers = {}
dialogue_ids = sorted(
    (path.name for path in DATA_DIR.iterdir() if path.is_dir()),
    key=numeric_key,
)
for dialogue_id in dialogue_ids:
    dialogue_dir = DATA_DIR / dialogue_id
    chunk_paths = sorted(dialogue_dir.glob("chunk_*.wav"), key=chunk_key)
    audios = [read_wave(path) for path in chunk_paths]
    transcripts = transcribe(audios)
    order = order_dialogue(transcripts, prefixes[dialogue_id])
    if sorted(order) != list(range(len(chunk_paths))):
        raise ValueError("invalid predicted order")
    answers[dialogue_id] = to_ranks(order)

OUTPUT_PATH.write_text(
    json.dumps(answers, ensure_ascii=False, separators=(",", ":")),
    encoding="utf-8",
)
