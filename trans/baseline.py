import os
import random
import pandas as pd
import sentencepiece as spm
import sacrebleu
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import zipfile
from tqdm import tqdm

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


# Data paths
TRAIN_SRC = "dataset/train/train.zh"
TRAIN_TGT = "dataset/train/train.vi"
TEST_SRC = "dataset/test/test.zh"
SAVE_DIR = "./checkpoints"
SPM_ZH_PREFIX = os.path.join(SAVE_DIR, "spm_zh")
SPM_VI_PREFIX = os.path.join(SAVE_DIR, "spm_vi")

# Model hyperparameters
VOCAB_SIZE = 3000
EMB_SIZE = 64
HID_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.01
MAX_LEN = 80
SEED = 42

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Set random seeds for reproducibility
os.makedirs(SAVE_DIR, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def read_lines(path):
    """Read lines from a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

# Load training and test data
train_src = read_lines(TRAIN_SRC)
train_tgt = read_lines(TRAIN_TGT)
test_src = read_lines(TEST_SRC)

print(f"Training samples: {len(train_src)}")
print(f"Test samples: {len(test_src)}")
print(f"\nExample Chinese sentence: {train_src[0]}")
print(f"Example Vietnamese sentence: {train_tgt[0]}")


def train_spm(input_file, model_prefix, vocab_size=VOCAB_SIZE):
    """Train a SentencePiece BPE model."""
    args = (
        f"--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} "
        "--model_type=bpe --character_coverage=1.0 "
        "--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3"
    )
    spm.SentencePieceTrainer.Train(args)
    print(f"Trained SentencePiece model: {model_prefix}.model")

def load_sp(model_path):
    """Load a trained SentencePiece model."""
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp


# Train Chinese tokenizer
tmp_zh = os.path.join(SAVE_DIR, "tmp_zh.txt")
if not os.path.exists(SPM_ZH_PREFIX + ".model"):
    with open(tmp_zh, "w", encoding="utf-8") as f:
        for s in train_src:
            f.write(s + "\n")
    train_spm(tmp_zh, SPM_ZH_PREFIX)

# Train Vietnamese tokenizer
tmp_vi = os.path.join(SAVE_DIR, "tmp_vi.txt")
if not os.path.exists(SPM_VI_PREFIX + ".model"):
    with open(tmp_vi, "w", encoding="utf-8") as f:
        for s in train_tgt:
            f.write(s + "\n")
    train_spm(tmp_vi, SPM_VI_PREFIX)

# Load tokenizers
sp_zh = load_sp(SPM_ZH_PREFIX + ".model")
sp_vi = load_sp(SPM_VI_PREFIX + ".model")

print(f"\nChinese vocab size: {sp_zh.GetPieceSize()}")
print(f"Vietnamese vocab size: {sp_vi.GetPieceSize()}")

# Test tokenization
test_sent = train_src[0]
tokens = sp_zh.EncodeAsIds(test_sent)
print(f"\nExample tokenization:")
print(f"Original: {test_sent}")
print(f"Token IDs: {tokens[:20]}...")


class TranslationDataset(Dataset):
    """Dataset for Chinese-Vietnamese translation pairs."""
    
    def __init__(self, src, tgt, sp_src, sp_tgt, max_len=MAX_LEN):
        self.src = src
        self.tgt = tgt
        self.sp_src = sp_src
        self.sp_tgt = sp_tgt
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        # Add BOS (2) and EOS (3) tokens
        src_ids = [2] + self.sp_src.EncodeAsIds(self.src[idx])[:self.max_len-2] + [3]
        tgt_ids = [2] + self.sp_tgt.EncodeAsIds(self.tgt[idx])[:self.max_len-2] + [3]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    """Collate function to pad sequences to the same length."""
    srcs, tgts = zip(*batch)
    max_src = max(len(s) for s in srcs)
    max_tgt = max(len(t) for t in tgts)
    
    # Pad with 0 (PAD token)
    src_pad = torch.zeros(len(batch), max_src, dtype=torch.long)
    tgt_pad = torch.zeros(len(batch), max_tgt, dtype=torch.long)
    
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, :len(s)] = s
        tgt_pad[i, :len(t)] = t
    
    return src_pad, tgt_pad


# Create training dataset and dataloader
# Split data: 90% train, 10% validation
total_samples = len(train_src)
train_size = int(0.9 * total_samples)

train_src_split = train_src[:train_size]
train_tgt_split = train_tgt[:train_size]
valid_src = train_src[train_size:]
valid_tgt = train_tgt[train_size:]

print(f"Total samples: {total_samples}")
print(f"Training samples: {len(train_src_split)}")
print(f"Validation samples: {len(valid_src)}")

dataset = TranslationDataset(train_src_split, train_tgt_split, sp_zh, sp_vi)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

valid_dataset = TranslationDataset(valid_src, valid_tgt, sp_zh, sp_vi)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

print(f"Training batches: {len(dataloader)}")
print(f"Validation batches: {len(valid_loader)}")


class EncoderRNN(nn.Module):
    """GRU-based encoder."""
    
    def __init__(self, vocab_size, emb_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.dropout = nn.Dropout(0.3)
        self.rnn = nn.GRU(emb_size, hidden_size, batch_first=True)

    def forward(self, src):
        emb = self.dropout(self.embedding(src))
        _, hidden = self.rnn(emb)
        return hidden


class DecoderRNN(nn.Module):
    """GRU-based decoder with teacher forcing."""
    
    def __init__(self, vocab_size, emb_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.dropout = nn.Dropout(0.3)
        self.rnn = nn.GRU(emb_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_step, hidden):
        emb = self.dropout(self.embedding(input_step))
        output, hidden = self.rnn(emb, hidden)
        pred = self.fc(output.squeeze(1))
        return pred, hidden


class Seq2Seq(nn.Module):
    """Sequence-to-sequence model."""
    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.3):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.fc.out_features
        
        # Store outputs
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(src.device)
        
        # Encode source sentence
        hidden = self.encoder(src)
        
        # Start with BOS token
        input_step = tgt[:, 0].unsqueeze(1)
        
        # Decode step by step
        for t in range(1, tgt_len):
            pred, hidden = self.decoder(input_step, hidden)
            outputs[:, t] = pred
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            input_step = tgt[:, t].unsqueeze(1) if teacher_force else pred.argmax(1).unsqueeze(1)
        
        return outputs


# Initialize model
encoder = EncoderRNN(sp_zh.GetPieceSize(), EMB_SIZE, HID_SIZE)
decoder = DecoderRNN(sp_vi.GetPieceSize(), EMB_SIZE, HID_SIZE)
model = Seq2Seq(encoder, decoder).to(DEVICE)

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model has {count_parameters(model):,} trainable parameters")
print(f"\nModel architecture:")
print(model)


def train_epoch(model, dataloader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for src, tgt in dataloader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(src, tgt)
        
        # Calculate loss (ignore first BOS token)
        loss = criterion(
            output[:, 1:].reshape(-1, output.size(-1)), 
            tgt[:, 1:].reshape(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_bleu(model, dataloader, sp_tgt):
    """Evaluate model using SacreBLEU metric."""
    model.eval()
    hyps, refs = [], []
    
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    for src, tgt in pbar:
        src = src.to(DEVICE)
        
        # Encode
        hidden = model.encoder(src)
        
        # Decode (greedy)
        input_step = torch.full((src.size(0), 1), 2, dtype=torch.long, device=DEVICE)
        decoded = [[] for _ in range(src.size(0))]
        
        for _ in range(MAX_LEN):
            pred, hidden = model.decoder(input_step, hidden)
            next_token = pred.argmax(1).unsqueeze(1)
            
            for i in range(src.size(0)):
                decoded[i].append(next_token[i].item())
            
            input_step = next_token
        
        # Convert to text
        for i in range(src.size(0)):
            ids = decoded[i]
            if 3 in ids:  # Stop at EOS
                ids = ids[:ids.index(3)]
            hyps.append(sp_tgt.DecodeIds(ids))
            
            ref_ids = tgt[i].tolist()[1:-1]  # Remove BOS and EOS
            refs.append(sp_tgt.DecodeIds([x for x in ref_ids if x not in [0, 1]]))
    
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return bleu.score


# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
optimizer = optim.Adam(model.parameters(), lr=LR)

print("Starting training...\n")

# Training loop with best model saving
best_bleu = 0.0
best_model_path = os.path.join(SAVE_DIR, "best_model.pt")

for epoch in range(1, EPOCHS + 1):
    loss = train_epoch(model, dataloader, criterion, optimizer)
    bleu = evaluate_bleu(model, valid_loader, sp_vi)
    
    # Save best model
    if bleu > best_bleu:
        best_bleu = bleu
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'bleu': bleu,
            'loss': loss
        }, best_model_path)
        print(f"Epoch {epoch:02d} | Loss={loss:.3f} | SacreBLEU={bleu:.2f} Best model saved!")
    else:
        print(f"Epoch {epoch:02d} | Loss={loss:.3f} | SacreBLEU={bleu:.2f}")

print(f"\nTraining completed!")
print(f"Best validation BLEU: {best_bleu:.2f}")
print(f"Best model saved to: {best_model_path}")

# Load best model for inference
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']}")


@torch.no_grad()
def translate_test(model, test_src, sp_src, sp_tgt, out_path):
    """Translate test set and save to CSV."""
    model.eval()
    outputs = []
    
    for s in tqdm(test_src, desc="Translating"):
        # Tokenize source sentence
        src_ids = [2] + sp_src.EncodeAsIds(s)[:MAX_LEN-2] + [3]
        src_tensor = torch.tensor(src_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        
        # Encode
        hidden = model.encoder(src_tensor)
        
        # Decode
        input_step = torch.full((1, 1), 2, dtype=torch.long, device=DEVICE)
        decoded = []
        
        for _ in range(MAX_LEN):
            pred, hidden = model.decoder(input_step, hidden)
            next_token = pred.argmax(1)
            token = next_token.item()
            
            if token == 3:  # EOS
                break
            
            decoded.append(token)
            input_step = next_token.unsqueeze(1)
        
        # Decode to Vietnamese text
        vi_sent = sp_tgt.DecodeIds(decoded)
        outputs.append(vi_sent)
    
    # Save to CSV
    df = pd.DataFrame({"tieng_trung": test_src, "tieng_viet": outputs})
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    
    return out_path


# Translate test set
submission_path = os.path.join("submission.csv")
translate_test(model, test_src, sp_zh, sp_vi, submission_path)

print(f"Translation completed!")
print(f"Saved to: {submission_path}")

# Display sample translations
df_result = pd.read_csv(submission_path)
print("\nSample translations:")
print(df_result.head(10))


# Create ZIP file for submission
zip_path = os.path.join("submission.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(submission_path, arcname="submission.csv")

print(f"Submission ZIP created: {zip_path}")
print(f"File size: {os.path.getsize(zip_path) / 1024:.2f} KB")
