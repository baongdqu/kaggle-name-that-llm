import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, diags
from sklearn.svm import LinearSVC
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
import re, time, gc

t0 = time.time()
print("🚀 [Step 1] Loading Datasets and Deep Learning OOFs...", flush=True)

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

# DL blend: Q14B (1.0) + Q8B (1.0) + DeB (1.5)
dl_blend = (1.0 * m1 + 1.0 * m2 + 1.5 * m3) / 3.5
log_dl = np.log(dl_blend + 1e-12)

all_pids = train['prompt_id'].values
unique_pids = np.unique(all_pids)

print("\n🛠️ [Step 2] Building Multi-View Stylometric Feature Matrix...", flush=True)
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
    """Subtract each group's mean row (sparse); groups outside [lo, hi] are left as they are."""
    X = csr_matrix(X, dtype=np.float32)
    _, inv, cnt = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    keep = np.flatnonzero((cnt >= lo) & (cnt <= hi))
    if not len(keep):
        return X.copy()
    col = np.full(len(cnt), -1)
    col[keep] = np.arange(len(keep))
    rows = np.flatnonzero(col[inv] >= 0)
    M = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, col[inv][rows])), shape=(X.shape[0], len(keep)), dtype=np.float32)
    means = diags(np.array(1.0 / cnt[keep], dtype=np.float32)) @ (M.T @ X)
    return (X - M @ means).tocsr()

t_c0 = time.time()
X_centered = sparse_group_centre(X_all, all_pids)
print(f"✅ Sparse Group Centering completed in {time.time()-t_c0:.3f}s! Total Features: {X_all.shape[1]}", flush=True)

print("\n🧠 [Step 3] Precomputing Leak-Free 5-Fold Dual-SVC and Family Gates...", flush=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_svc_plain = np.zeros((len(train), 18), dtype=np.float32)
oof_svc_centered = np.zeros((len(train), 18), dtype=np.float32)

FAMILIES = {
    'claude': [m for m in LABELS if m.startswith('claude')],
    'gemini': [m for m in LABELS if m.startswith('gemini')],
    'gpt':    [m for m in LABELS if m.startswith('gpt')],
    'glm':    [m for m in LABELS if m.startswith('glm')]
}

oof_gates = {f: np.zeros((len(train), len(m)), dtype=np.float32) for f, m in FAMILIES.items()}

for fold, (tr_pids_idx, val_pids_idx) in enumerate(kf.split(unique_pids)):
    t_f = time.time()
    tr_pids = unique_pids[tr_pids_idx]
    val_pids = unique_pids[val_pids_idx]
    tr_idx = np.where(np.isin(all_pids, tr_pids))[0]
    val_idx = np.where(np.isin(all_pids, val_pids))[0]
    
    # 1. Plain Global SVC
    svc_p = LinearSVC(C=0.5, random_state=42, max_iter=2000, dual=True)
    svc_p.fit(X_all[tr_idx], y_true[tr_idx])
    dp = svc_p.decision_function(X_all[val_idx])
    exp_dp = np.exp(dp - dp.max(axis=1, keepdims=True))
    oof_svc_plain[val_idx] = np.log(exp_dp / exp_dp.sum(axis=1, keepdims=True) + 1e-12)
    
    # 2. Centered Global SVC (Prompt-Invariant Author Signal)
    # Groups are split strictly by prompt_id so slicing global centered matrix is 100% leak-free
    X_tr_c = X_centered[tr_idx]
    X_val_c = X_centered[val_idx]
    
    svc_c = LinearSVC(C=0.5, random_state=42, max_iter=2000, dual=True)
    svc_c.fit(X_tr_c, y_true[tr_idx])
    dc = svc_c.decision_function(X_val_c)
    exp_dc = np.exp(dc - dc.max(axis=1, keepdims=True))
    oof_svc_centered[val_idx] = np.log(exp_dc / exp_dc.sum(axis=1, keepdims=True) + 1e-12)
    
    # 3. Family Specialized Gates (Trained on centered features)
    for fname, fmodels in FAMILIES.items():
        f_indices = [L2I[m] for m in fmodels]
        sub_tr = [i for i, row in enumerate(tr_idx) if y_true[row] in f_indices]
        sub_y = [f_indices.index(y_true[tr_idx[i]]) for i in sub_tr]
        
        gate_svc = LinearSVC(C=1.0, random_state=42, max_iter=2000, dual=True)
        gate_svc.fit(X_tr_c[sub_tr], sub_y)
        d_gate = gate_svc.decision_function(X_val_c)
        if d_gate.ndim == 1:
            d_gate = np.column_stack([-d_gate, d_gate])
        oof_gates[fname][val_idx] = d_gate
        
    gc.collect()
    print(f"  Fold {fold+1}/5 completed in {time.time()-t_f:.1f}s", flush=True)

def apply_family_gate(LOGIT, gate_scores, family_name, lam=1.0):
    """
    Confidence-aware within-family re-ranking preserving total family mass.
    """
    idx = [L2I[m] for m in FAMILIES[family_name]]
    score = gate_scores - gate_scores.mean(axis=1, keepdims=True)
    scale = np.sqrt(np.mean(score * score, axis=1, keepdims=True))
    score = score / np.maximum(scale, 0.25)
    
    # Gate log-probs
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
    
    # Family mass
    b_max = base.max(axis=1, keepdims=True)
    family_mass = b_max.squeeze() + np.log(np.exp(base - b_max).sum(axis=1))
    
    LOGIT[:, idx] = family_mass[:, None] + mixed_lp
    return LOGIT

print("\n📊 [Step 4] Evaluating SOTA Blend Combinations...", flush=True)

def run_evaluation(w_plain, w_cent, w_gate, temp=0.5):
    preds = np.zeros(len(train), dtype=int)
    tot = log_dl + w_plain * oof_svc_plain + w_cent * oof_svc_centered
    
    if w_gate > 0:
        tot = tot.copy()
        for fname in FAMILIES.keys():
            tot = apply_family_gate(tot, oof_gates[fname], fname, lam=w_gate)
            
    for p in unique_pids:
        idx = np.where(all_pids == p)[0]
        M = tot[idx]
        
        # Soft-Sinkhorn balancing
        K = np.exp(M / temp)
        for _ in range(15):
            K = K / (K.sum(axis=1, keepdims=True) + 1e-12)
            K = K / (K.sum(axis=0, keepdims=True) + 1e-12)
            
        r, c = linear_sum_assignment(-K)
        preds[idx[r]] = c
        
    return balanced_accuracy_score(y_true, preds)

print(f"👉 Baseline DL only:                            {run_evaluation(0.0, 0.0, 0.0)*100:.2f}%")
print(f"👉 DL + Plain SVC (w=2.0):                      {run_evaluation(2.0, 0.0, 0.0)*100:.2f}%")
print(f"👉 DL + Centered SVC (w=2.0):                   {run_evaluation(0.0, 2.0, 0.0)*100:.2f}%")
print(f"👉 Dual-SVC (Plain 1.5 + Centered 1.5):         {run_evaluation(1.5, 1.5, 0.0)*100:.2f}%")
print(f"👉 Dual-SVC + Family Gate (w_gate=1.0):         {run_evaluation(1.5, 1.5, 1.0)*100:.2f}%")
print(f"👉 Dual-SVC + Family Gate (w_gate=2.0):         {run_evaluation(1.5, 1.5, 2.0)*100:.2f}%")

best_score = 0
best_params = None
for wp in [1.0, 1.5, 2.0, 2.5]:
    for wc in [1.0, 1.5, 2.0]:
        for wg in [0.5, 1.0, 1.5, 2.0]:
            for tmp in [0.35, 0.5, 0.7, 1.0]:
                acc = run_evaluation(wp, wc, wg, temp=tmp)
                if acc > best_score:
                    best_score = acc
                    best_params = (wp, wc, wg, tmp)
                    print(f"🔥 BEST: {best_score*100:.2f}% | wp={wp}, wc={wc}, wg={wg}, tmp={tmp}", flush=True)

print("\n" + "="*60)
print(f"🏆 ULTIMATE OPTIMIZED CV SCORE: {best_score*100:.2f}% with params {best_params}")
print("="*60, flush=True)
