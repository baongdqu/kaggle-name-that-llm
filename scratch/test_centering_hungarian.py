import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
import re, time

t0 = time.time()
train = pd.read_csv('data/train.csv')

p1 = r'scratch\out_tokattn1\tokattn_oof.npz'
p2 = r'scratch\out_tokattn2\tokattn_q8l22_oof.npz'
p3 = r'scratch\out_tokattn2\tokattn_debl23_oof.npz'

d1 = np.load(p1, allow_pickle=True)
d2 = np.load(p2, allow_pickle=True)
d3 = np.load(p3, allow_pickle=True)

LABELS = ['claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
 'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash','glm-5.3','glm-5.3-flash',
 'gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra','kimi-k3','qwen3.8-max']
L2I = {l: i for i, l in enumerate(LABELS)}
y_true = np.array([L2I[m] for m in train['model'].values])

idx_map1 = [list(d1['classes']).index(l) for l in LABELS]
idx_map2 = [list(d2['classes']).index(l) for l in LABELS]
idx_map3 = [list(d3['classes']).index(l) for l in LABELS]

dl_blend = (d1['probs'][:, idx_map1] + d2['probs'][:, idx_map2] + d3['probs'][:, idx_map3]) / 3.0
log_dl = np.log(dl_blend + 1e-12)

# Multi-view TFIDF
texts = train['text'].astype(str).tolist()
_FW = set('a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little'.split())
_shape = lambda t: re.sub(r'[^\W\d_]+', lambda m_: 'X' if m_[0].isupper() else 'x', t)
_func = lambda t: ' '.join(w.lower() if w.lower() in _FW or not w.isalnum() else 'CONTENT' for w in re.findall(r'\w+|[^\w\s]', t))

matrices = []
for view, lo, hi, mf, wv in [
    ('char', 3, 5, 30000, 1.0),
    ('word', 1, 3, 20000, 0.7),
    ('shape', 2, 6, 10000, 0.5),
    ('function', 1, 3, 10000, 0.5)
]:
    prep = _shape if view == 'shape' else _func if view == 'function' else (lambda s_: s_)
    an = 'char' if view == 'shape' else 'word' if view == 'function' else view
    o = dict(analyzer=an, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf, dtype=np.float32)
    if an == 'word': o['token_pattern'] = r'(?u)\b\w+\b|[^\w\s]'
    v = TfidfVectorizer(**o)
    matrices.append(v.fit_transform([prep(t) for t in texts]) * wv)

X_all = hstack(matrices, format='csr')

all_pids = train['prompt_id'].values
unique_pids = np.unique(all_pids)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_svc = np.zeros((len(train), 18), dtype=np.float32)
for fold, (tr_pids_idx, val_pids_idx) in enumerate(kf.split(unique_pids)):
    tr_pids = unique_pids[tr_pids_idx]
    val_pids = unique_pids[val_pids_idx]
    tr_idx = np.where(np.isin(all_pids, tr_pids))[0]
    val_idx = np.where(np.isin(all_pids, val_pids))[0]
    
    svc = LinearSVC(C=0.5, random_state=42, max_iter=2000, dual=True)
    svc.fit(X_all[tr_idx], y_true[tr_idx])
    d_val = svc.decision_function(X_all[val_idx])
    d_m = d_val.max(axis=1, keepdims=True)
    exp_d = np.exp(d_val - d_m)
    oof_svc[val_idx] = np.log(exp_d / exp_d.sum(axis=1, keepdims=True) + 1e-12)

# Raw uncentered baseline
base_mat = log_dl + 2.0 * oof_svc
preds_raw = np.zeros(len(train), dtype=int)
for pid in unique_pids:
    idx = np.where(all_pids == pid)[0]
    cost = -base_mat[idx]
    r, c = linear_sum_assignment(cost)
    preds_raw[idx[r]] = c
print(f"👉 Raw Base Score (DL + 2.0*LinearSVC): {balanced_accuracy_score(y_true, preds_raw)*100:.2f}%")

# Test 1: Column Centering (Subtract prompt-mean for each class)
preds_col = np.zeros(len(train), dtype=int)
for pid in unique_pids:
    idx = np.where(all_pids == pid)[0]
    M = base_mat[idx].copy()
    # Subtract mean across rows for each column (relative model advantage)
    M_centered = M - M.mean(axis=0, keepdims=True)
    cost = -M_centered
    r, c = linear_sum_assignment(cost)
    preds_col[idx[r]] = c
acc_col = balanced_accuracy_score(y_true, preds_col)
print(f"👉 Column-Centered Hungarian: {acc_col*100:.2f}% (diff: {(acc_col - balanced_accuracy_score(y_true, preds_raw))*100:+.2f}%)")

# Test 2: Double Centering (Bipartite Centering)
preds_dbl = np.zeros(len(train), dtype=int)
for pid in unique_pids:
    idx = np.where(all_pids == pid)[0]
    M = base_mat[idx].copy()
    row_mean = M.mean(axis=1, keepdims=True)
    col_mean = M.mean(axis=0, keepdims=True)
    grand_mean = M.mean()
    M_dbl = M - row_mean - col_mean + grand_mean
    cost = -M_dbl
    r, c = linear_sum_assignment(cost)
    preds_dbl[idx[r]] = c
acc_dbl = balanced_accuracy_score(y_true, preds_dbl)
print(f"👉 Double-Centered Hungarian: {acc_dbl*100:.2f}% (diff: {(acc_dbl - balanced_accuracy_score(y_true, preds_raw))*100:+.2f}%)")

# Test 3: Soft-Sinkhorn Normalization (Alternating Row/Column Softmax)
def sinkhorn_knopp(M, num_iters=20, temp=0.5):
    K = np.exp(M / temp)
    for _ in range(num_iters):
        K = K / (K.sum(axis=1, keepdims=True) + 1e-12)
        K = K / (K.sum(axis=0, keepdims=True) + 1e-12)
    return K

for temp in [0.2, 0.35, 0.5, 0.7, 1.0]:
    preds_sink = np.zeros(len(train), dtype=int)
    for pid in unique_pids:
        idx = np.where(all_pids == pid)[0]
        M = base_mat[idx].copy()
        P = sinkhorn_knopp(M, num_iters=30, temp=temp)
        cost = -P
        r, c = linear_sum_assignment(cost)
        preds_sink[idx[r]] = c
    acc_s = balanced_accuracy_score(y_true, preds_sink)
    print(f"👉 Sinkhorn (temp={temp:.2f}) + Hungarian: {acc_s*100:.2f}%")

