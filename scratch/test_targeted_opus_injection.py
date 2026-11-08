import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, diags
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

_FW = set('a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little'.split())
_shape = lambda t: re.sub(r'[^\W\d_]+', lambda m_: 'X' if m_[0].isupper() else 'x', t)
_func = lambda t: ' '.join(w.lower() if w.lower() in _FW or not w.isalnum() else 'CONTENT' for w in re.findall(r'\w+|[^\w\s]', t))

matrices = []
for view, lo, hi, mf, wv in [
    ('char', 3, 5, 35000, 1.0),
    ('char_wb', 2, 4, 25000, 0.8),
    ('word', 1, 3, 20000, 0.7),
    ('shape', 2, 6, 10000, 0.5),
    ('function', 1, 3, 10000, 0.5)
]:
    prep = _shape if view == 'shape' else _func if view == 'function' else (lambda s_: s_)
    an = 'char' if view == 'shape' else 'word' if view == 'function' else ('char_wb' if view == 'char_wb' else view)
    o = dict(analyzer=an, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf, dtype=np.float32)
    if an == 'word': o['token_pattern'] = r'(?u)\b\w+\b|[^\w\s]'
    v = TfidfVectorizer(**o)
    matrices.append(v.fit_transform([prep(t) for t in texts]) * wv)

X_all = hstack(matrices, format='csr')

def sparse_group_centre(X, groups, lo=2, hi=18):
    X = csr_matrix(X, dtype=np.float32)
    _, inv, cnt = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    keep = np.flatnonzero((cnt >= lo) & (cnt <= hi))
    if not len(keep): return X.copy()
    col = np.full(len(cnt), -1); col[keep] = np.arange(len(keep))
    rows = np.flatnonzero(col[inv] >= 0)
    M = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, col[inv][rows])), shape=(X.shape[0], len(keep)), dtype=np.float32)
    means = diags(np.array(1.0 / cnt[keep], dtype=np.float32)) @ (M.T @ X)
    return (X - M @ means).tocsr()

X_centered = sparse_group_centre(X_all, all_pids)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_svc_plain = np.zeros((len(train), 18), dtype=np.float32)
oof_svc_centered = np.zeros((len(train), 18), dtype=np.float32)

for fold, (tr_pids_idx, val_pids_idx) in enumerate(kf.split(unique_pids)):
    tr_pids = unique_pids[tr_pids_idx]
    val_pids = unique_pids[val_pids_idx]
    tr_idx = np.where(np.isin(all_pids, tr_pids))[0]
    val_idx = np.where(np.isin(all_pids, val_pids))[0]
    
    svc_p = LinearSVC(C=0.5, random_state=42, max_iter=2000, dual=True)
    svc_p.fit(X_all[tr_idx], y_true[tr_idx])
    dp = svc_p.decision_function(X_all[val_idx])
    exp_dp = np.exp(dp - dp.max(axis=1, keepdims=True))
    oof_svc_plain[val_idx] = np.log(exp_dp / exp_dp.sum(axis=1, keepdims=True) + 1e-12)
    
    svc_c = LinearSVC(C=0.5, random_state=42, max_iter=2000, dual=True)
    svc_c.fit(X_centered[tr_idx], y_true[tr_idx])
    dc = svc_c.decision_function(X_centered[val_idx])
    exp_dc = np.exp(dc - dc.max(axis=1, keepdims=True))
    oof_svc_centered[val_idx] = np.log(exp_dc / exp_dc.sum(axis=1, keepdims=True) + 1e-12)

# Train 71.11% Opus Gate
claude_ext = pd.read_csv('data/ntl_extra_passages_claude.csv')
opus_models = ['claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5']
sub_ext_opus = claude_ext[claude_ext['model'].isin(opus_models)].copy().reset_index(drop=True)
y_opus_train = np.array([opus_models.index(m) for m in sub_ext_opus['model'].values])

v_opus1 = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=35000, sublinear_tf=True, dtype=np.float32)
v_opus2 = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', max_features=25000, sublinear_tf=True, dtype=np.float32)

X_op_tr = hstack([v_opus1.fit_transform(sub_ext_opus['text'].values), v_opus2.fit_transform(sub_ext_opus['text'].values)], format='csr')
X_op_va = hstack([v_opus1.transform(texts), v_opus2.transform(texts)], format='csr')

clf_opus = LinearSVC(C=0.5, random_state=42, max_iter=2000, dual=True)
clf_opus.fit(X_op_tr, y_opus_train)
gate_opus_logits = clf_opus.decision_function(X_op_va)

# Base 66.31% Dual-SVC
base_tot = log_dl + 1.5 * oof_svc_plain + 1.5 * oof_svc_centered

def eval_matching(tot_matrix, temp=0.5):
    preds = np.zeros(len(train), dtype=int)
    for p in unique_pids:
        idx = np.where(all_pids == p)[0]
        M = tot_matrix[idx]
        K = np.exp(M / temp)
        for _ in range(15):
            K = K / (K.sum(axis=1, keepdims=True) + 1e-12)
            K = K / (K.sum(axis=0, keepdims=True) + 1e-12)
        r, c = linear_sum_assignment(-K)
        preds[idx[r]] = c
    return balanced_accuracy_score(y_true, preds)

print(f"👉 Baseline Dual-SVC (5 views): {eval_matching(base_tot)*100:.2f}%", flush=True)

# Test targeted injection on Opus columns
opus_indices = [L2I[m] for m in opus_models]

for lam in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
    mod_tot = base_tot.copy()
    if lam > 0:
        # Standardize gate logits
        sc = gate_opus_logits - gate_opus_logits.mean(axis=1, keepdims=True)
        scale = np.maximum(sc.std(axis=1, keepdims=True), 0.25)
        sc = sc / scale
        
        # Add to opus columns
        mod_tot[:, opus_indices] += lam * sc
        
    score = eval_matching(mod_tot, temp=0.5)
    print(f"  lam={lam:.1f} -> Balanced Accuracy: {score*100:.2f}%", flush=True)

