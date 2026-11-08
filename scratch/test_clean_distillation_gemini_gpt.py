import pandas as pd, numpy as np, re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("🚀 Testing Clean In-Domain Data Distillation for Gemini Flash & GPT-5.6...", flush=True)

df_ext = pd.read_parquet('data/external_short_wide.parquet')
train = pd.read_csv('data/train.csv')

print(f"Total rows in external: {len(df_ext)}")

# Define strict cleaning filter
refusal_patterns = [
    'i cannot', 'i can not', "i can't", 'as an ai', 'i am unable',
    'against my safety', 'i apologize', 'my programming', 'responsible ai',
    'as a helpful', 'i must refuse', 'content policy'
]

def clean_filter(text):
    if not isinstance(text, str): return False
    words = text.split()
    if len(words) < 45 or len(words) > 250: return False
    t_lower = text.lower()
    for pat in refusal_patterns:
        if pat in t_lower[:150]: return False
    # Check that text is predominantly English prose
    alpha_chars = sum(c.isalpha() for c in text)
    if alpha_chars / max(1, len(text)) < 0.70: return False
    return True

# 1. GPT-5.6 Sibling Distillation
gpt_models = ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra']
# Find prompts where all 3 GPT-5.6 models responded cleanly
gpt_clean_mask = df_ext[gpt_models[0]].apply(clean_filter) & \
                 df_ext[gpt_models[1]].apply(clean_filter) & \
                 df_ext[gpt_models[2]].apply(clean_filter)

clean_gpt_prompts = df_ext[gpt_clean_mask].copy().reset_index(drop=True)
print(f"Clean matched GPT-5.6 triplets: {len(clean_gpt_prompts)} prompts ({len(clean_gpt_prompts)*3} samples)")

# Build external training set
ext_gpt_rows = []
for m in gpt_models:
    sub_df = clean_gpt_prompts[['prompt', m]].copy()
    sub_df['model'] = m
    sub_df.rename(columns={m: 'text'}, inplace=True)
    ext_gpt_rows.append(sub_df)
df_ext_gpt = pd.concat(ext_gpt_rows, ignore_index=True)

# Official validation set (450 rows)
sub_val_gpt = train[train['model'].isin(gpt_models)].copy().reset_index(drop=True)

y_tr_gpt = np.array([gpt_models.index(m) for m in df_ext_gpt['model'].values])
y_va_gpt = np.array([gpt_models.index(m) for m in sub_val_gpt['model'].values])

# Vectorize
v_word = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', max_features=25000, sublinear_tf=True)
v_char = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=35000, sublinear_tf=True)

X_tr_gpt = hstack([v_word.fit_transform(df_ext_gpt['text']), v_char.fit_transform(df_ext_gpt['text'])], format='csr')
X_va_gpt = hstack([v_word.transform(sub_val_gpt['text']), v_char.transform(sub_val_gpt['text'])], format='csr')

# Evaluate transfer
clf_gpt = LogisticRegression(C=0.5, max_iter=1000)
clf_gpt.fit(X_tr_gpt, y_tr_gpt)
preds_gpt = clf_gpt.predict(X_va_gpt)
acc_gpt = accuracy_score(y_va_gpt, preds_gpt)

print(f"👉 Distilled Clean GPT-5.6 3-Way Zero-Shot Transfer: {acc_gpt*100:.2f}% (vs 33.33% random baseline) 🔥")

# Compare with raw uncleaned external transfer (was 35.78%)
print(f"   Improvement over raw uncleaned external: {acc_gpt*100 - 35.78:+.2f}%")
