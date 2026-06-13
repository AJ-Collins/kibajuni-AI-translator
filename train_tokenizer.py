"""
Step 2: Train BPE tokenizers on the Bajuni and English corpus.
Trains two tokenizers:
  - bajuni_tokenizer/   (target language — most important)
  - english_tokenizer/  (source language)
"""

import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from pathlib import Path

CORPUS_CSV   = r"C:\Users\ashik\Desktop\Photogrammetry\project\AITranslation\corpus_clean.csv"
OUTPUT_DIR   = r"C:\Users\ashik\Desktop\Photogrammetry\project\AITranslation\tokenizers"

# BPE vocab size — keep small given limited data; increase when corpus grows
VOCAB_SIZE   = 4000
MIN_FREQUENCY = 2   # ignore tokens appearing fewer than 2 times


def build_tokenizer(texts: list[str], vocab_size: int, min_freq: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Wrap with BOS/EOS post-processing
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)],
    )
    return tokenizer


def report_tokenizer(name: str, tokenizer: Tokenizer, samples: list[str]):
    print(f"\n  [{name}]")
    print(f"    Vocab size : {tokenizer.get_vocab_size()}")
    token_lengths = [len(tokenizer.encode(t).ids) for t in samples[:200]]
    avg = sum(token_lengths) / len(token_lengths)
    print(f"    Avg tokens per sentence (first 200): {avg:.1f}")
    print(f"    Sample encoding of first sentence:")
    enc = tokenizer.encode(samples[0])
    print(f"      Text   : {samples[0]}")
    print(f"      Tokens : {enc.tokens}")
    print(f"      IDs    : {enc.ids}")


def main():
    df = pd.read_csv(CORPUS_CSV)
    en_texts  = df["english"].dropna().astype(str).tolist()
    baj_texts = df["bajuni"].dropna().astype(str).tolist()

    print(f"\nTraining on {len(en_texts)} English / {len(baj_texts)} Bajuni sentences")
    print(f"Vocab size target : {VOCAB_SIZE}  |  Min frequency : {MIN_FREQUENCY}")

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    print("\n── Training Bajuni tokenizer ──")
    baj_tok = build_tokenizer(baj_texts, VOCAB_SIZE, MIN_FREQUENCY)
    baj_tok.save(str(out / "bajuni_tokenizer.json"))

    print("\n── Training English tokenizer ──")
    en_tok  = build_tokenizer(en_texts, VOCAB_SIZE, MIN_FREQUENCY)
    en_tok.save(str(out / "english_tokenizer.json"))

    print("\n" + "="*55)
    print("  TOKENIZER REPORT")
    print("="*55)
    report_tokenizer("Bajuni",  baj_tok, baj_texts)
    report_tokenizer("English", en_tok,  en_texts)

    print(f"\n  Saved to: {OUTPUT_DIR}")
    print("="*55)

    # Quick fertility check — tokens per word (lower = better compression)
    print("\n  Fertility check (tokens per word):")
    for name, tok, texts in [("Bajuni", baj_tok, baj_texts), ("English", en_tok, en_texts)]:
        words  = sum(len(t.split()) for t in texts[:200])
        tokens = sum(len(tok.encode(t).ids) for t in texts[:200])
        print(f"    {name}: {tokens/words:.2f} tokens/word  (ideal < 2.0 for well-trained tokenizer)")


if __name__ == "__main__":
    main()