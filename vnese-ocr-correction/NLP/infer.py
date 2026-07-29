# infer.py

import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

import sentencepiece as spm

# =========================================================
# CONFIG
# =========================================================

TEST_CSV = "test.csv"

MAX_LEN = 64

EMBED_DIM = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# =========================================================
# LOAD TOKENIZER
# =========================================================

sp = spm.SentencePieceProcessor()
sp.load("bpe.model")

VOCAB_SIZE = sp.vocab_size()

PAD_ID = sp.pad_id()
UNK_ID = sp.unk_id()
BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()

print("Tokenizer loaded")
print("Vocab size:", VOCAB_SIZE)

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

# =========================================================
# LOAD MODEL
# =========================================================

model = Seq2Seq().to(DEVICE)

checkpoint = torch.load(
    "best.pt",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded")

# =========================================================
# ENCODE
# =========================================================

def encode(text):

    ids = sp.encode(text)

    ids = [BOS_ID] + ids + [EOS_ID]

    ids = ids[:MAX_LEN]

    return ids

# =========================================================
# GREEDY DECODE
# =========================================================

def decode_greedy(text):

    src = encode(text)

    src = torch.tensor(src).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        hidden = model.encoder(src)

        token = torch.tensor([BOS_ID]).to(DEVICE)

        generated = []

        for _ in range(MAX_LEN):

            output, hidden = model.decoder(
                token,
                hidden
            )

            token = output.argmax(1)

            pred_id = token.item()

            if pred_id == EOS_ID:
                break

            generated.append(pred_id)

    return sp.decode(generated)

# =========================================================
# LOAD TEST
# =========================================================

df = pd.read_csv(TEST_CSV)

print("Test samples:", len(df))

# =========================================================
# INFERENCE
# =========================================================

predictions = []

print("\nStarting inference...\n")

for text in tqdm(
    df["input"],
    desc="Infer"
):

    pred = decode_greedy(str(text))

    predictions.append(pred)

print("\nInference completed")

# =========================================================
# SAVE SUBMISSION
# =========================================================

submission = pd.DataFrame({
    "id": df["id"],
    "corrected_text": predictions
})

submission.to_csv(
    "submission.csv",
    index=False
)

print("\nSaved submission.csv")
print(submission.head())