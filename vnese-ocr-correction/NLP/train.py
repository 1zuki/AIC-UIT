# train.py

import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sentencepiece as spm

# =========================================================
# CONFIG
# =========================================================

TRAIN_CSV = "train.csv"

SP_MODEL_PREFIX = "bpe"
VOCAB_SIZE = 4000

MAX_LEN = 64

BATCH_SIZE = 128

EMBED_DIM = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.2

LR = 1e-3
EPOCHS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42

# =========================================================
# SEED
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(TRAIN_CSV)

df = df.dropna()

df["input"] = df["input"].astype(str).str.strip()
df["corrected_text"] = df["corrected_text"].astype(str).str.strip()

df = df[
    (df["input"] != "") &
    (df["corrected_text"] != "")
]

df = df.drop_duplicates(
    subset=["input", "corrected_text"]
)

print("Dataset size:", len(df))

# =========================================================
# TRAIN TOKENIZER
# =========================================================

with open("spm_corpus.txt", "w", encoding="utf-8") as f:

    for text in df["input"]:
        f.write(text + "\n")

    for text in df["corrected_text"]:
        f.write(text + "\n")

spm.SentencePieceTrainer.train(
    input="spm_corpus.txt",
    model_prefix=SP_MODEL_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type="bpe",
    character_coverage=1.0,

    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)

sp = spm.SentencePieceProcessor()
sp.load(f"{SP_MODEL_PREFIX}.model")

VOCAB_SIZE = sp.vocab_size()

PAD_ID = sp.pad_id()
UNK_ID = sp.unk_id()
BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()

print("Vocab size:", VOCAB_SIZE)

# =========================================================
# DATASET
# =========================================================

class TextDataset(Dataset):

    def __init__(self, df):

        self.inputs = df["input"].tolist()
        self.targets = df["corrected_text"].tolist()

    def encode(self, text):

        ids = sp.encode(text)

        ids = [BOS_ID] + ids + [EOS_ID]

        ids = ids[:MAX_LEN]

        return ids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        src = self.encode(self.inputs[idx])
        tgt = self.encode(self.targets[idx])

        return {
            "src": src,
            "tgt": tgt
        }

def collate_fn(batch):

    srcs = [x["src"] for x in batch]
    tgts = [x["tgt"] for x in batch]

    src_max = max(len(x) for x in srcs)
    tgt_max = max(len(x) for x in tgts)

    padded_src = []
    padded_tgt = []

    for s in srcs:

        s = s + [PAD_ID] * (src_max - len(s))
        padded_src.append(s)

    for t in tgts:

        t = t + [PAD_ID] * (tgt_max - len(t))
        padded_tgt.append(t)

    return {
        "src": torch.tensor(padded_src),
        "tgt": torch.tensor(padded_tgt)
    }

dataset = TextDataset(df)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# =========================================================
# MODEL
# =========================================================

class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBED_DIM,
            padding_idx=PAD_ID
        )

        self.gru = nn.GRU(
            EMBED_DIM,
            HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(
            HIDDEN_SIZE * 2,
            HIDDEN_SIZE
        )

    def forward(self, x):

        emb = self.embedding(x)

        outputs, hidden = self.gru(emb)

        hidden = torch.cat(
            (hidden[-2], hidden[-1]),
            dim=1
        )

        hidden = torch.tanh(
            self.fc(hidden)
        )

        hidden = hidden.unsqueeze(0)

        return hidden

class Decoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBED_DIM,
            padding_idx=PAD_ID
        )

        self.gru = nn.GRU(
            EMBED_DIM,
            HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True
        )

        self.fc = nn.Linear(
            HIDDEN_SIZE,
            VOCAB_SIZE
        )

    def forward(self, input_token, hidden):

        input_token = input_token.unsqueeze(1)

        emb = self.embedding(input_token)

        output, hidden = self.gru(
            emb,
            hidden
        )

        output = output.squeeze(1)

        pred = self.fc(output)

        return pred, hidden

class Seq2Seq(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, src, tgt, teacher_forcing=0.5):

        B = src.shape[0]
        T = tgt.shape[1]

        outputs = torch.zeros(
            B,
            T,
            VOCAB_SIZE
        ).to(DEVICE)

        hidden = self.encoder(src)

        input_token = tgt[:, 0]

        for t in range(1, T):

            output, hidden = self.decoder(
                input_token,
                hidden
            )

            outputs[:, t] = output

            top1 = output.argmax(1)

            use_teacher = random.random() < teacher_forcing

            input_token = tgt[:, t] if use_teacher else top1

        return outputs

model = Seq2Seq().to(DEVICE)

# =========================================================
# TRAIN
# =========================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_ID
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

best_loss = 1e9

os.makedirs("checkpoints", exist_ok=True)

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    pbar = tqdm(loader)

    for batch in pbar:

        src = batch["src"].to(DEVICE)
        tgt = batch["tgt"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(src, tgt)

        outputs = outputs[:, 1:].reshape(-1, VOCAB_SIZE)
        targets = tgt[:, 1:].reshape(-1)

        loss = criterion(outputs, targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

        pbar.set_description(
            f"epoch={epoch+1} loss={loss.item():.4f}"
        )

    avg_loss = total_loss / len(loader)

    print("\n" + "=" * 50)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Average Loss: {avg_loss:.6f}")
    print("=" * 50)

    # SAVE EPOCH CHECKPOINT

    checkpoint_path = f"checkpoints/epoch_{epoch+1}.pt"

    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss
    }, checkpoint_path)

    print(f"Saved: {checkpoint_path}")

    # SAVE LAST

    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss
    }, "last.pt")

    # SAVE BEST

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        }, "best.pt")

        print(f"New best model saved! Loss = {avg_loss:.6f}")

print("\nTraining completed.")