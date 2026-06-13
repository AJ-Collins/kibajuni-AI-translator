"""
Flask web server for the English → Kibajuni translator.
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn as nn
from pathlib import Path
from tokenizers import Tokenizer
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE         = Path(r"C:\Users\ashik\Desktop\Photogrammetry\project\AITranslation")
TOK_DIR      = BASE / "tokenizers"
MODEL_PATH   = BASE / "model_en_bajuni" / "best_model.pt"
SRC_TOK_PATH = TOK_DIR / "english_tokenizer.json"
TGT_TOK_PATH = TOK_DIR / "bajuni_tokenizer.json"
MAX_LEN      = 64
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ── Model ──────────────────────────────────────────────────────────────────
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

# ── Beam search ────────────────────────────────────────────────────────────
def beam_decode(model, src_ids, src_mask, tok, beam_size=4, max_len=MAX_LEN, lp=0.6):
    model.eval()
    with torch.no_grad():
        mem, mem_kpm = model.encode(src_ids, src_mask)
        beams, completed = [(0.0, [tok.tgt_bos])], []
        for _ in range(max_len):
            candidates = []
            for score, seq in beams:
                if seq[-1] == tok.tgt_eos:
                    completed.append((score, seq)); continue
                dec    = torch.tensor([seq], dtype=torch.long, device=device)
                logits = model.decode_step(dec, mem, mem_kpm)
                log_probs = torch.log_softmax(logits[0, -1], dim=-1)
                topk   = log_probs.topk(beam_size)
                for lprob, tid in zip(topk.values.tolist(), topk.indices.tolist()):
                    candidates.append((score + lprob, seq + [tid]))
            if not candidates: break
            candidates.sort(key=lambda x: x[0] / (len(x[1]) ** lp), reverse=True)
            beams = candidates[:beam_size]
            if all(s[-1] == tok.tgt_eos for _, s in beams):
                completed.extend(beams); break
        completed = completed or beams
        completed.sort(key=lambda x: x[0] / (len(x[1]) ** lp), reverse=True)
        return completed[0][1][1:]

# ── Load model once at startup ─────────────────────────────────────────────
print(f"Loading model on {device}...")
tok   = Tokenizers()
model = TranslatorModel(tok.src_vsz, tok.tgt_vsz).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print("Model ready.")

# ── Flask app ──────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(BASE / "web"), static_url_path="")

@app.route("/")
def index():
    return send_from_directory(str(BASE / "web"), "index.html")

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        ids, mask = tok.encode_src([text])
        out       = beam_decode(model, ids.to(device), mask.to(device), tok)
        result    = tok.decode(out)
        return jsonify({"translation": result, "source": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    web_dir = BASE / "web"
    web_dir.mkdir(exist_ok=True)
    print(f"\nOpen your browser at: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)