"""
Step 3 (v3): English → Bajuni Seq2Seq
Uses EncoderDecoderModel (avoids MarianMT masking bug in new transformers).
Custom BPE tokenizers, GPU fp16, early stopping.
"""

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import evaluate
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
BASE         = Path(r"C:\Users\ashik\Desktop\Photogrammetry\project\AITranslation")
CSV          = BASE / "corpus_clean.csv"
TOK_DIR      = BASE / "tokenizers"
OUT_DIR      = BASE / "model_en_bajuni"
SRC_TOK_PATH = TOK_DIR / "english_tokenizer.json"
TGT_TOK_PATH = TOK_DIR / "bajuni_tokenizer.json"

# ── Hyperparameters ────────────────────────────────────────────────────────
MAX_LEN    = 64
BATCH_SIZE = 32
EPOCHS     = 80
LR         = 3e-4
VAL_SPLIT  = 0.1
SEED       = 42
PATIENCE   = 10     # early stopping

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Tokenizer ──────────────────────────────────────────────────────────────
class Tokenizers:
    def __init__(self):
        self.src = Tokenizer.from_file(str(SRC_TOK_PATH))
        self.tgt = Tokenizer.from_file(str(TGT_TOK_PATH))
        for t in (self.src, self.tgt):
            t.enable_truncation(max_length=MAX_LEN)

        self.src_pad = self.src.token_to_id("[PAD]")
        self.tgt_pad = self.tgt.token_to_id("[PAD]")
        self.tgt_bos = self.tgt.token_to_id("[BOS]")
        self.tgt_eos = self.tgt.token_to_id("[EOS]")
        self.src_vsz = self.src.get_vocab_size()
        self.tgt_vsz = self.tgt.get_vocab_size()

        print(f"\nSrc vocab={self.src_vsz}  Tgt vocab={self.tgt_vsz}")
        print(f"PAD src={self.src_pad} tgt={self.tgt_pad}  BOS={self.tgt_bos}  EOS={self.tgt_eos}")

    def encode_src(self, texts):
        encs = self.src.encode_batch(texts)
        ids  = [e.ids for e in encs]
        # manual pad to MAX_LEN
        ids  = [seq + [self.src_pad]*(MAX_LEN-len(seq)) for seq in ids]
        ids  = torch.tensor(ids, dtype=torch.long)
        mask = (ids != self.src_pad).long()
        return ids, mask

    def encode_tgt(self, texts):
        encs = self.tgt.encode_batch(texts)
        ids  = [e.ids for e in encs]
        ids  = [seq + [self.tgt_pad]*(MAX_LEN-len(seq)) for seq in ids]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        skip = {self.tgt_pad, self.tgt_bos, self.tgt_eos}
        ids  = [i for i in ids if i not in skip]
        return self.tgt.decode(ids)

# ── Dataset ────────────────────────────────────────────────────────────────
class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tok: Tokenizers):
        self.src = src_texts
        self.tgt = tgt_texts
        self.tok = tok

    def __len__(self): return len(self.src)

    def __getitem__(self, idx):
        src_ids, src_mask = self.tok.encode_src([self.src[idx]])
        tgt_ids           = self.tok.encode_tgt([self.tgt[idx]])[0]

        # decoder input: shift right (BOS + tokens[:-1])
        dec_input = torch.cat([
            torch.tensor([self.tok.tgt_bos]),
            tgt_ids[:-1]
        ])

        # labels: tokens (the decoder predicts next token)
        labels = tgt_ids.clone()
        labels[labels == self.tok.tgt_pad] = -100

        return {
            "src_ids":   src_ids[0],
            "src_mask":  src_mask[0],
            "dec_input": dec_input,
            "labels":    labels,
        }

# ── Model ──────────────────────────────────────────────────────────────────
class TranslatorModel(nn.Module):
    def __init__(self, src_vsz, tgt_vsz, d_model=256, nhead=4,
                 num_enc=4, num_dec=4, ffn=512, dropout=0.2, max_len=MAX_LEN):
        super().__init__()
        self.src_emb = nn.Embedding(src_vsz, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vsz, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len + 1, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ffn, dropout, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, ffn, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_enc)
        self.decoder = nn.TransformerDecoder(dec_layer, num_dec)
        self.proj    = nn.Linear(d_model, tgt_vsz)
        self.d_model = d_model
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _pos(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.pos_emb(pos)

    def encode(self, src_ids, src_mask):
        x    = self.src_emb(src_ids) + self._pos(src_ids)
        # TransformerEncoder expects key_padding_mask where True = ignore
        kpm  = (src_mask == 0)
        mem  = self.encoder(x, src_key_padding_mask=kpm)
        return mem, kpm

    def decode_step(self, tgt_ids, memory, mem_kpm):
        t    = tgt_ids.size(1)
        x    = self.tgt_emb(tgt_ids) + self._pos(tgt_ids)
        caus = nn.Transformer.generate_square_subsequent_mask(t, device=x.device)
        out  = self.decoder(x, memory, tgt_mask=caus,
                            memory_key_padding_mask=mem_kpm)
        return self.proj(out)   # [B, T, tgt_vsz]

    def forward(self, src_ids, src_mask, dec_input):
        mem, mem_kpm = self.encode(src_ids, src_mask)
        return self.decode_step(dec_input, mem, mem_kpm)

# ── Greedy decode for BLEU ─────────────────────────────────────────────────
def greedy_decode(model, src_ids, src_mask, tok: Tokenizers, max_len=MAX_LEN):
    model.eval()
    with torch.no_grad():
        mem, mem_kpm = model.encode(src_ids, src_mask)
        dec = torch.full((src_ids.size(0), 1), tok.tgt_bos, dtype=torch.long, device=src_ids.device)
        for _ in range(max_len):
            logits = model.decode_step(dec, mem, mem_kpm)  # [B,T,V]
            nxt    = logits[:, -1, :].argmax(-1, keepdim=True)
            dec    = torch.cat([dec, nxt], dim=1)
            if (nxt == tok.tgt_eos).all():
                break
    return dec[:, 1:]  # strip BOS

# ── Training ───────────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, scheduler, scaler, criterion, train=True):
    model.train() if train else model.eval()
    total_loss, total_tok = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            src_ids   = batch["src_ids"].to(device)
            src_mask  = batch["src_mask"].to(device)
            dec_input = batch["dec_input"].to(device)
            labels    = batch["labels"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type=="cuda")):
                logits = model(src_ids, src_mask, dec_input)  # [B,T,V]
                B, T, V = logits.shape
                loss = criterion(logits.reshape(B*T, V), labels.reshape(B*T))

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            valid = (labels != -100).sum().item()
            total_loss += loss.item() * valid
            total_tok  += valid

    return total_loss / max(total_tok, 1)

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(CSV).dropna(subset=["english","bajuni"])
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    split    = int(len(df) * (1 - VAL_SPLIT))
    train_df = df.iloc[:split]
    val_df   = df.iloc[split:]
    print(f"\nTrain: {len(train_df)}  Val: {len(val_df)}")

    tok      = Tokenizers()
    train_ds = TranslationDataset(train_df["english"].tolist(), train_df["bajuni"].tolist(), tok)
    val_ds   = TranslationDataset(val_df["english"].tolist(),   val_df["bajuni"].tolist(),   tok)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TranslatorModel(tok.src_vsz, tok.tgt_vsz).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.1f}M")

    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=LR,
                           steps_per_epoch=len(train_dl), epochs=EPOCHS, pct_start=0.1)
    scaler    = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    bleu_metric = evaluate.load("sacrebleu")

    best_bleu, patience_count = 0.0, 0

    print(f"\n{'Epoch':>5} {'TrainLoss':>10} {'ValLoss':>9} {'BLEU':>6}")
    print("─" * 38)

    for epoch in range(1, EPOCHS + 1):
        tr_loss = run_epoch(model, train_dl, optimizer, scheduler, scaler, criterion, train=True)
        vl_loss = run_epoch(model, val_dl,   optimizer, scheduler, scaler, criterion, train=False)

        # BLEU on val set
        preds, refs = [], []
        for batch in val_dl:
            src_ids  = batch["src_ids"].to(device)
            src_mask = batch["src_mask"].to(device)
            out_ids  = greedy_decode(model, src_ids, src_mask, tok)
            for ids in out_ids.cpu().tolist():
                preds.append(tok.decode(ids))
        refs = [[tok.decode([t for t in batch["labels"][i].tolist() if t != -100])] for batch in val_dl for i in range(len(batch["labels"]))]

        bleu = bleu_metric.compute(predictions=preds, references=refs)["score"]
        print(f"{epoch:>5} {tr_loss:>10.4f} {vl_loss:>9.4f} {bleu:>6.2f}")

        if bleu > best_bleu:
            best_bleu = bleu
            patience_count = 0
            torch.save(model.state_dict(), OUT_DIR / "best_model.pt")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}. Best BLEU: {best_bleu:.2f}")
                break

    # ── Sample translations ───────────────────────────────────────────────
    print(f"\n── Sample translations (best model, BLEU={best_bleu:.2f}) ──")
    model.load_state_dict(torch.load(OUT_DIR / "best_model.pt", weights_only=True))
    model.eval()
    for i in range(min(5, len(val_df))):
        src = val_df["english"].iloc[i]
        ref = val_df["bajuni"].iloc[i]
        ids, mask = tok.encode_src([src])
        out  = greedy_decode(model, ids.to(device), mask.to(device), tok)
        pred = tok.decode(out[0].cpu().tolist())
        print(f"\n  EN  : {src}")
        print(f"  REF : {ref}")
        print(f"  PRED: {pred}")

    print(f"\n✅  Model saved → {OUT_DIR / 'best_model.pt'}")

if __name__ == "__main__":
    main()