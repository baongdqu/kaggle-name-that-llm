import numpy as np, pandas as pd, re, time
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, diags
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

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

dl_blend = (1.0 * d1['probs'][:, idx_map1] + 1.0 * d2['probs'][:, idx_map2] + 1.5 * d3['probs'][:, idx_map3]) / 3.5
log_dl = np.log(dl_blend + 1e-12)

all_pids = train['prompt_id'].values
unique_pids = np.unique(all_pids)
texts = train['text'].astype(str).tolist()

print("⚡ [1] Testing Sibling Gates: Rank-Order + N-Gram Hybrid...", flush=True)

# Rank features
raw_f = []
raw_f.append(np.array([len(t) for t in texts], dtype=np.float32))
raw_f.append(np.array([len(t.split()) for t in texts], dtype=np.float32))
puncs = [':', ';', '—', '–', '-', '“', '”', '’', '‘', '(', ')', '/', '!', '?', '`', '*', '"', "'", '.', ',']
for p in puncs:
    raw_f.append(np.array([t.count(p) for t in texts], dtype=np.float32))
raw_f.append(np.array([sum(1 for c in t if c.isupper()) / max(1, len(t)) for t in texts], dtype=np.float32))
raw_f.append(np.array([sum(1 for c in t if c.isdigit()) / max(1, len(t)) for t in texts], dtype=np.float32))
raw_f.append(np.array([t.count('\n') for t in texts], dtype=np.float32))
raw_mat = np.column_stack(raw_f)

rank_mat = np.zeros_like(raw_mat)
for pid in unique_pids:
    idx = np.where(all_pids == pid)[0]
    p_data = raw_mat[idx]
    p_ranks = p_data.argsort(axis=0).argsort(axis=0).astype(np.float32) / (len(idx) - 1.0)
    rank_mat[idx] = p_ranks - 0.5 # zero-center rank in [-0.5, 0.5]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Benchmark Gemini 3.7 vs 3.8: Char_wb (4-6) + Rank Features
i_g37 = L2I['gemini-3.7-flash']
i_g38 = L2I['gemini-3.8-flash']
gem_idx = np.where(np.isin(y_true, [i_g37, i_g38]))[0]
y_gem = (y_true[gem_idx] == i_g38).astype(int)

v_gem = TfidfVectorizer(ngram_range=(4, 6), analyzer='char_wb', max_features=15000, sublinear_tf=True, dtype=np.float32)
X_gem_tfidf = v_gem.fit_transform([texts[i] for i in gem_idx])
X_gem_rank = csr_matrix(rank_mat[gem_idx] * 2.0)
X_gem_hybrid = hstack([X_gem_tfidf, X_gem_rank], format='csr')

oof_gem_hybrid = np.zeros(len(gem_idx))
for tr_p, va_p in kf.split(unique_pids):
    tr_pids = unique_pids[tr_p]
    va_pids = unique_pids[va_p]
    tr_i = np.where(np.isin(all_pids[gem_idx], tr_pids))[0]
    va_i = np.where(np.isin(all_pids[gem_idx], va_pids))[0]
    
    clf = LogisticRegression(C=1.0, max_iter=500)
    clf.fit(X_gem_hybrid[tr_i], y_gem[tr_i])
    oof_gem_hybrid[va_i] = clf.predict_proba(X_gem_hybrid[va_i])[:, 1]

acc_gem = balanced_accuracy_score(y_gem, (oof_gem_hybrid >= 0.5).astype(int))
print(f"🔥 Gemini 3.7 vs 3.8 [Char_wb + Rank Hybrid]: {acc_gem*100:.2f}% (Previous best was 58.00%)", flush=True)

# Benchmark GPT Sol vs Terra: Word(1-2) + Char(3-5) + Rank Features
i_sol = L2I['gpt-5.6-sol']
i_terra = L2I['gpt-5.6-terra']
gpt_idx = np.where(np.isin(y_true, [i_sol, i_terra]))[0]
y_gpt = (y_true[gpt_idx] == i_terra).astype(int)

v_gpt_w = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', max_features=10000, sublinear_tf=True, dtype=np.float32)
v_gpt_c = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=15000, sublinear_tf=True, dtype=np.float32)
X_gpt_tfidf = hstack([v_gpt_w.fit_transform([texts[i] for i in gpt_idx]), v_gpt_c.fit_transform([texts[i] for i in gpt_idx])], format='csr')
X_gpt_rank = csr_matrix(rank_mat[gpt_idx] * 2.0)
X_gpt_hybrid = hstack([X_gpt_tfidf, X_gpt_rank], format='csr')

oof_gpt_hybrid = np.zeros(len(gpt_idx))
for tr_p, va_p in kf.split(unique_pids):
    tr_pids = unique_pids[tr_p]
    va_pids = unique_pids[va_p]
    tr_i = np.where(np.isin(all_pids[gpt_idx], tr_pids))[0]
    va_i = np.where(np.isin(all_pids[gpt_idx], va_pids))[0]
    
    clf = LogisticRegression(C=1.0, max_iter=500)
    clf.fit(X_gpt_hybrid[tr_i], y_gpt[tr_i])
    oof_gpt_hybrid[va_i] = clf.predict_proba(X_gpt_hybrid[va_i])[:, 1]

acc_gpt = balanced_accuracy_score(y_gpt, (oof_gpt_hybrid >= 0.5).astype(int))
print(f"🔥 GPT Sol vs Terra [TF-IDF + Rank Hybrid]: {acc_gpt*100:.2f}% (Previous best was 58.33%)", flush=True)
