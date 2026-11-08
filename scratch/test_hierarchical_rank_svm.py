import numpy as np, pandas as pd, re, time
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, diags
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

print("🚀 Testing Hierarchical Sibling Decomposition + Prompt Rank Order Features...", flush=True)

train = pd.read_csv('data/train.csv')
p1 = r'scratch\out_tokattn1\tokattn_oof.npz'
p2 = r'scratch\out_tokattn2\tokattn_q8l22_oof.npz'
p3 = r'scratch\out_tokattn2\tokattn_debl23_oof.npz'

d1 = np.load(p1, allow_pickle=True)
d2 = np.load(p2, allow_pickle=True)
d3 = np.load(p3, allow_pickle=True)

LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
L2I = {l: i for i, l in enumerate(LABELS)}
y_true = np.array([L2I[m] for m in train['model'].values])

idx_map1 = [list(d1['classes']).index(l) for l in LABELS]
idx_map2 = [list(d2['classes']).index(l) for l in LABELS]
idx_map3 = [list(d3['classes']).index(l) for l in LABELS]

m1 = d1['probs'][:, idx_map1]
m2 = d2['probs'][:, idx_map2]
m3 = d3['probs'][:, idx_map3]

dl_blend = (1.0 * m1 + 1.0 * m2 + 1.5 * m3) / 3.5
log_dl = np.log(dl_blend + 1e-12)

all_pids = train['prompt_id'].values
unique_pids = np.unique(all_pids)
texts = train['text'].astype(str).tolist()

# 1. Compute Within-Prompt Rank Order Features
print("📊 [1] Computing Within-Prompt Rank Order Matrix...", flush=True)
features_raw = []

# Character length & word length
chars_len = np.array([len(t) for t in texts], dtype=np.float32)
words_len = np.array([len(t.split()) for t in texts], dtype=np.float32)
features_raw.append(chars_len)
features_raw.append(words_len)

# Punctuation counts
puncs = [':', ';', '—', '–', '-', '“', '”', '’', '‘', '(', ')', '/', '!', '?', '`', '*', '"', "'", '.', ',']
for p in puncs:
    features_raw.append(np.array([t.count(p) for t in texts], dtype=np.float32))

# Ratios
features_raw.append(np.array([sum(1 for c in t if c.isupper()) / max(1, len(t)) for t in texts], dtype=np.float32))
features_raw.append(np.array([sum(1 for c in t if c.isdigit()) / max(1, len(t)) for t in texts], dtype=np.float32))
features_raw.append(np.array([t.count('\n') for t in texts], dtype=np.float32))

raw_matrix = np.column_stack(features_raw)

# Convert to within-prompt rank (0 to 17 normalized to [0, 1])
rank_matrix = np.zeros_like(raw_matrix)
for pid in unique_pids:
    idx = np.where(all_pids == pid)[0]
    p_data = raw_matrix[idx]
    # Argsort rank
    p_ranks = p_data.argsort(axis=0).argsort(axis=0).astype(np.float32) / (len(idx) - 1.0)
    rank_matrix[idx] = p_ranks

print(f"✅ Rank Matrix Shape: {rank_matrix.shape}", flush=True)

# 2. Check pairwise accuracy on sister pairs using Rank Features alone
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, m_a, m_b in [
    ('Gemini 3.7 vs 3.8', 'gemini-3.7-flash', 'gemini-3.8-flash'),
    ('GPT Sol vs Terra', 'gpt-5.6-sol', 'gpt-5.6-terra'),
    ('GLM 5.3 vs Flash', 'glm-5.3', 'glm-5.3-flash')
]:
    i_a = L2I[m_a]
    i_b = L2I[m_b]
    sub_idx = np.where(np.isin(y_true, [i_a, i_b]))[0]
    y_sub = (y_true[sub_idx] == i_b).astype(int)
    X_sub = rank_matrix[sub_idx]
    
    oof_preds = np.zeros(len(sub_idx))
    for tr_p, va_p in kf.split(unique_pids):
        tr_pids = unique_pids[tr_p]
        va_pids = unique_pids[va_p]
        tr_i = np.where(np.isin(all_pids[sub_idx], tr_pids))[0]
        va_i = np.where(np.isin(all_pids[sub_idx], va_pids))[0]
        
        clf = LogisticRegression(C=1.0)
        clf.fit(X_sub[tr_i], y_sub[tr_i])
        oof_preds[va_i] = clf.predict_proba(X_sub[va_i])[:, 1]
        
    acc = balanced_accuracy_score(y_sub, (oof_preds >= 0.5).astype(int))
    print(f"🎯 Rank-Order Pairwise Accuracy [{name}]: {acc*100:.2f}%", flush=True)

print("Finished fast rank evaluation!", flush=True)
