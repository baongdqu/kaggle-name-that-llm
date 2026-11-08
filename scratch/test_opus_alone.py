import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, diags
from sklearn.svm import LinearSVC
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
import time

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

v_char = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=35000, sublinear_tf=True, dtype=np.float32)
v_word = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', max_features=25000, sublinear_tf=True, dtype=np.float32)

X_all = hstack([v_char.fit_transform(texts), v_word.fit_transform(texts)], format='csr')

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
oof_svc_cent = np.zeros((len(train), 18), dtype=np.float32)

for fold, (tr_pids_idx, val_pids_idx) in enumerate(kf.split(unique_pids)):
    tr_pids = unique_pids[tr_pids_idx]
    val_pids = unique_pids[val_pids_idx]
    tr_idx = np.where(np.isin(all_pids, tr_pids))[0]
    val_idx = np.where(np.isin(all_pids, val_pids))[0]
    
    svc = LinearSVC(C=0.3, tol=1e-3, random_state=42, max_iter=1000, dual=True)
    svc.fit(X_centered[tr_idx], y_true[tr_idx])
    d_val = svc.decision_function(X_centered[val_idx])
    exp_d = np.exp(d_val - d_val.max(axis=1, keepdims=True))
    oof_svc_cent[val_idx] = np.log(exp_d / exp_d.sum(axis=1, keepdims=True) + 1e-12)

# Train Opus Specialist Gate on clean external 3,000 samples
claude_ext = pd.read_csv('data/ntl_extra_passages_claude.csv')
opus_models = ['claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5']
sub_ext_opus = claude_ext[claude_ext['model'].isin(opus_models)].copy().reset_index(drop=True)
y_opus_train = np.array([opus_models.index(m) for m in sub_ext_opus['model'].values])

v_opus = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=35000, sublinear_tf=True, dtype=np.float32)
X_opus_tr = v_opus.fit_transform(sub_ext_opus['text'].values)
X_opus_val = v_opus.transform(texts)

clf_opus = LinearSVC(C=0.5, tol=1e-3, random_state=42, max_iter=2000, dual=True)
clf_opus.fit(X_opus_tr, y_opus_train)
gate_opus_logits = clf_opus.decision_function(X_opus_val)

def apply_family_gate(LOGIT, gate_scores, fam_models, lam=1.0):
    idx = [L2I[m] for m in fam_models]
    score = gate_scores - gate_scores.mean(axis=1, keepdims=True)
    scale = np.sqrt(np.mean(score * score, axis=1, keepdims=True))
    score = score / np.maximum(scale, 0.25)
    
    exp_s = np.exp(score - score.max(axis=1, keepdims=True))
    gate_lp = np.log(exp_s / exp_s.sum(axis=1, keepdims=True) + 1e-12)
    
    base = LOGIT[:, idx].copy()
    exp_b = np.exp(base - base.max(axis=1, keepdims=True))
    base_lp = np.log(exp_b / exp_b.sum(axis=1, keepdims=True) + 1e-12)
    base_p = np.exp(base_lp)
    
    sp = np.sort(base_p, axis=1)
    margin = sp[:, -1] - sp[:, -2]
    confidence_gate = np.clip((0.45 - margin) / 0.45, 0.0, 1.0)
    strength = min(0.45, float(lam) / (1.0 + float(lam))) * confidence_gate
    
    mixed_lp = (1.0 - strength[:, None]) * base_lp + strength[:, None] * gate_lp
    exp_m = np.exp(mixed_lp - mixed_lp.max(axis=1, keepdims=True))
    mixed_lp = np.log(exp_m / exp_m.sum(axis=1, keepdims=True) + 1e-12)
    
    b_max = base.max(axis=1, keepdims=True)
    family_mass = b_max.squeeze() + np.log(np.exp(base - b_max).sum(axis=1))
    
    LOGIT[:, idx] = family_mass[:, None] + mixed_lp
    return LOGIT

def eval_model(w_svc=2.0, lam_opus=2.0, temp=0.5):
    tot = (log_dl + w_svc * oof_svc_cent).copy()
    if lam_opus > 0:
        tot = apply_family_gate(tot, gate_opus_logits, opus_models, lam=lam_opus)
    preds = np.zeros(len(train), dtype=int)
    for p in unique_pids:
        idx = np.where(all_pids == p)[0]
        M = tot[idx]
        K = np.exp(M / temp)
        for _ in range(15):
            K = K / (K.sum(axis=1, keepdims=True) + 1e-12)
            K = K / (K.sum(axis=0, keepdims=True) + 1e-12)
        r, c = linear_sum_assignment(-K)
        preds[idx[r]] = c
    return balanced_accuracy_score(y_true, preds)

acc_dl = eval_model(w_svc=0.0, lam_opus=0.0)
acc_dl_svc = eval_model(w_svc=2.0, lam_opus=0.0)
print(f"👉 Pure Deep Learning:     {acc_dl*100:.2f}%")
print(f"👉 DL + Centered Dual-SVC: {acc_dl_svc*100:.2f}%")

for w in [1.5, 2.0, 2.5, 3.0]:
    for l_op in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        for t in [0.3, 0.4, 0.5, 0.6]:
            sc = eval_model(w_svc=w, lam_opus=l_op, temp=t)
            if sc > 0.665:
                print(f"🚀 Score: {sc*100:.2f}% | w_svc={w}, lam_opus={l_op}, temp={t}")

