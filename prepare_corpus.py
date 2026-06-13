import pandas as pd

FILEPATH = r"C:\Users\ashik\Desktop\Photogrammetry\project\AITranslation\Elicitation_Prompts_EN_SW_PK_BAJU 5.6.2026.xlsx"

def load_sheet1(filepath):
    df = pd.read_excel(filepath, sheet_name="Sentence corpus ")
    pairs = df[["English", "Bajuni"]].copy()
    pairs.columns = ["english", "bajuni"]
    pairs["source"] = "sentence_corpus"
    return pairs

def load_sheet2(filepath):
    df = pd.read_excel(filepath, sheet_name=" AFS Sheet")
    # Only prompt pairs are populated; answer columns are empty
    pairs = df[["Prompt_EN", "Prompt in Kibajuni"]].copy()
    pairs.columns = ["english", "bajuni"]
    pairs["source"] = "afs_prompts"
    return pairs

def clean(df):
    df = df.dropna(subset=["english", "bajuni"])
    df["english"] = df["english"].astype(str).str.strip()
    df["bajuni"]  = df["bajuni"].astype(str).str.strip()
    # Drop empty strings
    df = df[(df["english"] != "") & (df["bajuni"] != "")]
    # Drop exact duplicates on the pair
    df = df.drop_duplicates(subset=["english", "bajuni"])
    return df

def report(df):
    print(f"\n{'='*55}")
    print(f"  PARALLEL CORPUS SUMMARY")
    print(f"{'='*55}")
    print(f"  Total usable pairs : {len(df)}")
    print(f"  From sentence_corpus: {(df.source == 'sentence_corpus').sum()}")
    print(f"  From afs_prompts    : {(df.source == 'afs_prompts').sum()}")
    print(f"\n  English avg length  : {df.english.str.len().mean():.0f} chars")
    print(f"  Bajuni avg length   : {df.bajuni.str.len().mean():.0f} chars")
    print(f"\n  Domain coverage (sentence_corpus):")
    # Re-load to get domain labels
    raw = pd.read_excel(FILEPATH, sheet_name="Sentence corpus ")
    for domain, grp in raw.groupby("DomainLabel"):
        print(f"    {len(grp):>4}  {domain}")
    print(f"\n  ⚠  Gap to 10k minimum: {max(0, 10000 - len(df))} more pairs needed")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    s1 = clean(load_sheet1(FILEPATH))
    s2 = clean(load_sheet2(FILEPATH))
    corpus = pd.concat([s1, s2], ignore_index=True)

    report(corpus)

    # Save clean corpus
    out = r"C:\Users\ashik\Desktop\Photogrammetry\project\AITranslation\corpus_clean.csv"
    corpus[["english", "bajuni", "source"]].to_csv(out, index=False, encoding="utf-8")
    print(f"  Saved → {out}")