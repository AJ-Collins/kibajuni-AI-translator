import pandas as pd
import sys

def explore_corpus(filepath):
    print(f"\n{'='*60}")
    print(f"FILE: {filepath}")
    print('='*60)

    # Load all sheets
    all_sheets = pd.read_excel(filepath, sheet_name=None)
    print(f"\nSheets found ({len(all_sheets)}): {list(all_sheets.keys())}")

    for sheet_name, df in all_sheets.items():
        print(f"\n{'─'*60}")
        print(f"SHEET: '{sheet_name}'")
        print(f"  Rows: {len(df)}  |  Columns: {len(df.columns)}")
        print(f"  Columns: {list(df.columns)}")

        # Null counts
        nulls = df.isnull().sum()
        if nulls.any():
            print(f"\n  Missing values:")
            for col, count in nulls[nulls > 0].items():
                pct = count / len(df) * 100
                print(f"    '{col}': {count} ({pct:.1f}%)")

        # Sample rows
        print(f"\n  First 3 rows:")
        print(df.head(3).to_string(index=False))

        # Check for likely language pair columns
        text_cols = df.select_dtypes(include='object').columns.tolist()
        if text_cols:
            print(f"\n  Text column stats:")
            for col in text_cols:
                sample = df[col].dropna()
                if len(sample) == 0:
                    continue
                avg_len = sample.astype(str).str.len().mean()
                unique = sample.nunique()
                print(f"    '{col}': avg_chars={avg_len:.0f}, unique={unique}/{len(df)}")

        # Duplicate check
        dupes = df.duplicated().sum()
        if dupes:
            print(f"\n  ⚠  Duplicate rows: {dupes}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Users\ashik\Desktop\Photogrammetry\project\AITranslation\Elicitation_Prompts_EN_SW_PK_BAJU 5.6.2026.xlsx"
    explore_corpus(path)