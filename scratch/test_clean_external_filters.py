import pandas as pd, numpy as np

df = pd.read_parquet('data/external_short_wide.parquet')
print(f"Total rows in external: {len(df)}")

gem_cols = ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash']
gpt_cols = ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra']

# Check non-null counts
for col in gem_cols + gpt_cols:
    cnt = df[col].notna().sum()
    print(f"{col}: {cnt} non-null samples")

# Check refusals
refusal_patterns = ['i cannot', 'i can not', "i can't", 'as an ai', 'i am unable', 'against my safety', 'i apologize']

def is_clean(text):
    if not isinstance(text, str): return False
    t = text.lower()
    if len(t.split()) < 30: return False  # too short
    for pat in refusal_patterns:
        if pat in t[:100]: return False  # refusal at start
    return True

clean_gemini = df[df[gem_cols[0]].apply(is_clean) & df[gem_cols[1]].apply(is_clean) & df[gem_cols[2]].apply(is_clean)]
print(f"\nClean Gemini Prompts (where all 3 flash models responded cleanly): {len(clean_gemini)}")

clean_gpt = df[df[gpt_cols[0]].apply(is_clean) & df[gpt_cols[1]].apply(is_clean) & df[gpt_cols[2]].apply(is_clean)]
print(f"Clean GPT-5.6 Prompts (where all 3 gpt-5.6 models responded cleanly): {len(clean_gpt)}")

# Sample a clean gemini text
print("\nSample Gemini 3.7 Flash:\n", clean_gemini['gemini-3.7-flash'].iloc[0][:300])
print("\nSample Gemini 3.8 Flash:\n", clean_gemini['gemini-3.8-flash'].iloc[0][:300])

