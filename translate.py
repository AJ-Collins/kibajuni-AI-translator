"""
Bajuni Translator — interactive inference
Usage:
    python translate.py                        # interactive mode
    python translate.py "The fish was caught"  # single sentence
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from tokenizers import Tokenizer

# ── Paths ──────────────────────────────────────────────────────────────────
BASE         = Path(r"C:\Users\ashik\Desktop\Photogrammetry\project\AITranslation")
TOK_DIR      = BASE / "tokenizers"
MODEL_PATH   = BASE / "model_en_bajuni" / "best_model.pt"
SRC_TOK_PATH = TOK_DIR / "english_tokenizer.json"
TGT_TOK_PATH = TOK_DIR / "bajuni_tokenizer.json"

MAX_LEN = 64
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def encode_src(self, texts):
        encs = self.src.encode_batch(texts)
        ids  = [e.ids + [self.src_pad] * (MAX_LEN - len(e.ids)) for e in encs]
        ids  = torch.tensor(ids, dtype=torch.long)
        mask = (ids != self.src_pad).long()
        return ids, mask

    def decode(self, ids):
        skip = {self.tgt_pad, self.tgt_bos, self.tgt_eos}
        return self.tgt.decode([i for i in ids if i not in skip])


# ── Model (must match train_model.py exactly) ──────────────────────────────
class TranslatorModel(nn.Module):
    def __init__(self, src_vsz, tgt_vsz, d_model=256, nhead=4,
                 num_enc=4, num_dec=4, ffn=512, dropout=0.2, max_len=MAX_LEN):
        super().__init__()
        self.src_emb = nn.Embedding(src_vsz, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vsz, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len + 1, d_model)
        enc_layer    = nn.TransformerEncoderLayer(d_model, nhead, ffn, dropout, batch_first=True)
        dec_layer    = nn.TransformerDecoderLayer(d_model, nhead, ffn, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_enc)
        self.decoder = nn.TransformerDecoder(dec_layer, num_dec)
        self.proj    = nn.Linear(d_model, tgt_vsz)

    def _pos(self, x):
        return self.pos_emb(torch.arange(x.size(1), device=x.device).unsqueeze(0))

    def encode(self, src_ids, src_mask):
        x   = self.src_emb(src_ids) + self._pos(src_ids)
        mem = self.encoder(x, src_key_padding_mask=(src_mask == 0))
        return mem, (src_mask == 0)

    def decode_step(self, tgt_ids, memory, mem_kpm):
        x    = self.tgt_emb(tgt_ids) + self._pos(tgt_ids)
        caus = nn.Transformer.generate_square_subsequent_mask(tgt_ids.size(1), device=x.device)
        out  = self.decoder(x, memory, tgt_mask=caus, memory_key_padding_mask=mem_kpm)
        return self.proj(out)


# ── Beam search (better than greedy for final output) ─────────────────────
def beam_decode(model, src_ids, src_mask, tok: Tokenizers,
                beam_size=4, max_len=MAX_LEN, length_penalty=0.6):
    model.eval()
    with torch.no_grad():
        mem, mem_kpm = model.encode(src_ids, src_mask)

        # Each beam: (score, token_ids)
        beams = [(0.0, [tok.tgt_bos])]
        completed = []

        for _ in range(max_len):
            candidates = []
            for score, seq in beams:
                if seq[-1] == tok.tgt_eos:
                    completed.append((score, seq))
                    continue
                dec = torch.tensor([seq], dtype=torch.long, device=device)
                logits = model.decode_step(dec, mem, mem_kpm)
                log_probs = torch.log_softmax(logits[0, -1], dim=-1)
                topk = log_probs.topk(beam_size)
                for lp, tok_id in zip(topk.values.tolist(), topk.indices.tolist()):
                    candidates.append((score + lp, seq + [tok_id]))

            if not candidates:
                break
            candidates.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
            beams = candidates[:beam_size]

            if all(s[-1] == tok.tgt_eos for _, s in beams):
                completed.extend(beams)
                break

        completed = completed or beams
        completed.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
        return completed[0][1][1:]  # strip BOS


# ── Main ───────────────────────────────────────────────────────────────────
def translate(sentences: list[str], tok: Tokenizers, model: TranslatorModel) -> list[str]:
    ids, mask = tok.encode_src(sentences)
    results   = []
    for i in range(len(sentences)):
        out  = beam_decode(model, ids[i:i+1].to(device), mask[i:i+1].to(device), tok)
        results.append(tok.decode(out))
    return results


def load_model(tok: Tokenizers) -> TranslatorModel:
    model = TranslatorModel(tok.src_vsz, tok.tgt_vsz).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    return model


def main():
    print(f"Loading model from {MODEL_PATH}...")
    tok   = Tokenizers()
    model = load_model(tok)
    print(f"Ready  ({device})\n")

    # Single sentence from command line
    if len(sys.argv) > 1:
        sentence = " ".join(sys.argv[1:])
        result   = translate([sentence], tok, model)[0]
        print(f"EN   : {sentence}")
        print(f"BAJU : {result}")
        return

    # Interactive mode
    print("English → Kibajuni translator")
    print("Type a sentence and press Enter. Type 'quit' to exit.\n")
    while True:
        try:
            text = input("EN > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        result = translate([text], tok, model)[0]
        print(f"BJ > {result}\n")


if __name__ == "__main__":
    main()