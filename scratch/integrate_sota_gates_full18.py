import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, diags
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
import time, gc

t0 = time.time()
print("🚀 [Step 1] Loading Datasets and Deep Learning Ensemble...", flush=True)

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

print("📚 [Step 2] Engineering Multi-View TF-IDF and Sparse Prompt Centering...", flush=True)
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

# Stylometric dense features
wlens = np.array([len(t.split()) for t in texts], dtype=np.float32)
rel_wlens = np.zeros(len(train), dtype=np.float32)
for p in unique_pids:
    idx = np.where(all_pids == p)[0]
    sw = wlens[idx]
    rel_wlens[idx] = (sw - sw.mean()) / max(1.0, sw.std())

colons = np.array([t.count(':') for t in texts], dtype=np.float32)
semis = np.array([t.count(';') for t in texts], dtype=np.float32)
curly = np.array([t.count('’') + t.count('“') for t in texts], dtype=np.float32)

print(f"✅ Features ready in {time.time()-t0:.1f}s. Precomputing 5-Fold Dual-SVC Base...", flush=True)

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

print("🎯 [Step 3] Training High-Precision Clean Specialist Gates...", flush=True)

# 1. Claude Opus Specialist Gate (Trained on 3,000 clean external samples)
claude_ext = pd.read_csv('data/ntl_extra_passages_claude.csv')
opus_models = ['claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5']
sub_ext_opus = claude_ext[claude_ext['model'].isin(opus_models)].copy().reset_index(drop=True)
y_opus_train = np.array([opus_models.index(m) for m in sub_ext_opus['model'].values])

v_opus = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=35000, sublinear_tf=True, dtype=np.float32)
X_opus_tr = v_opus.fit_transform(sub_ext_opus['text'].values)
X_opus_val = v_opus.transform(texts)

clf_opus = LinearSVC(C=0.5, tol=1e-3, random_state=42, max_iter=2000)
clf_opus.fit(X_opus_tr, y_opus_train)
gate_opus_logits = clf_opus.decision_function(X_opus_val)

# 2. Gemini Flash Specialist Gate (Trained with prompt-relative length and char n-grams)
gemini_flash_models = ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash']
gemini_idx = [L2I[m] for m in gemini_flash_models]
oof_gemini_gate = np.zeros((len(train), 3), dtype=np.float32)

for fold, (tr_pids_idx, val_pids_idx) in enumerate(kf.split(unique_pids)):
    tr_pids = unique_pids[tr_pids_idx]
    val_pids = unique_pids[val_pids_idx]
    tr_idx = np.where(np.isin(all_pids, tr_pids))[0]
    val_idx = np.where(np.isin(all_pids, val_pids))[0]
    
    sub_tr = [i for i, r in enumerate(tr_idx) if y_true[r] in gemini_idx]
    sub_y = [gemini_idx.index(y_true[tr_idx[i]]) for i in sub_tr]
    
    # Feature for gemini: char n-grams + relative length
    X_gem_tr = hstack([X_centered[tr_idx[sub_tr]], csr_matrix(rel_wlens[tr_idx[sub_tr]][:, None] * 3.0)], format='csr')
    X_gem_va = hstack([X_centered[val_idx], csr_matrix(rel_wlens[val_idx][:, None] * 3.0)], format='csr')
    
    clf_gem = LinearSVC(C=1.0, tol=1e-3, random_state=42, max_iter=2000)
    clf_gem.fit(X_gem_tr, sub_y)
    oof_gemini_gate[val_idx] = clf_gem.decision_function(X_gem_va)

# 3. GPT-5.6 Specialist Gate (Trained with punctuation features and char n-grams)
gpt56_models = ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra']
gpt56_idx = [L2I[m] for m in gpt56_models]
oof_gpt_gate = np.zeros((len(train), 3), dtype=np.float32)

for fold, (tr_pids_idx, val_pids_idx) in enumerate(kf.split(unique_pids)):
    tr_pids = unique_pids[tr_pids_idx]
    val_pids = unique_pids[val_pids_idx]
    tr_idx = np.where(np.isin(all_pids, tr_pids))[0]
    val_idx = np.where(np.isin(all_pids, val_pids))[0]
    
    sub_tr = [i for i, r in enumerate(tr_idx) if y_true[r] in gpt56_idx]
    sub_y = [gpt56_idx.index(y_true[tr_idx[i]]) for i in sub_tr]
    
    punc_feats_tr = np.column_stack([curly[tr_idx[sub_tr]], semis[tr_idx[sub_tr]], colons[tr_idx[sub_tr]]])
    punc_feats_va = np.column_stack([curly[val_idx], semis[val_idx], colons[val_idx]])
    
    X_gpt_tr = hstack([X_centered[tr_idx[sub_tr]], csr_matrix(punc_feats_tr * 2.0)], format='csr')
    X_gpt_va = hstack([X_centered[val_idx], csr_matrix(punc_feats_va * 2.0)], format='csr')
    
    clf_gpt = LinearSVC(C=1.0, tol=1e-3, random_state=42, max_iter=2000)
    clf_gpt.fit(X_gpt_tr, sub_y)
    oof_gpt_gate[val_idx] = clf_gpt.decision_function(X_gpt_va)

print(f"✅ All 3 Specialist Gates trained in {time.time()-t0:.1f}s!\n", flush=True)

print("📊 [Step 4] Integrating Gates & Evaluating 18-Class Balanced Accuracy...", flush=True)

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
    
    # Family mass conservation
    b_max = base.max(axis=1, keepdims=True)
    family_mass = b_max.squeeze() + np.log(np.exp(base - b_max).sum(axis=1))
    
    LOGIT[:, idx] = family_mass[:, None] + mixed_lp
    return LOGIT

def eval_integrated_model(w_svc=2.0, lam_opus=2.0, lam_gem=1.5, lam_gpt=1.5, temp=0.5):
    tot = log_dl + w_svc * oof_svc_cent
    tot = tot.copy()
    
    # Apply the 3 Specialist Gates
    if lam_opus > 0:
        tot = apply_family_gate(tot, gate_opus_logits, opus_models, lam=lam_opus)
    if lam_gem > 0:
        tot = apply_family_gate(tot, oof_gemini_gate, gemini_flash_models, lam=lam_gem)
    if lam_gpt > 0:
        tot = apply_family_gate(tot, oof_gpt_gate, gpt56_models, lam=lam_gpt)
        
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

acc_base_dl = eval_integrated_model(w_svc=0.0, lam_opus=0.0, lam_gem=0.0, lam_gpt=0.0)
acc_dl_svc = eval_integrated_model(w_svc=2.0, lam_opus=0.0, lam_gem=0.0, lam_gpt=0.0)
print(f"👉 Baseline Deep Learning (Q14B+Q8B+DeB):          {acc_base_dl*100:.2f}%")
print(f"👉 DL + Centered SVC:                             {acc_dl_svc*100:.2f}%")

best_score = 0
best_cfg = None

for w_s in [1.5, 2.0, 2.5]:
    for l_op in [1.0, 2.0, 3.0]:
        for l_gm in [1.0, 2.0]:
            for l_gp in [1.0, 2.0]:
                for t in [0.35, 0.5, 0.7]:
                    score = eval_integrated_model(w_svc=w_s, lam_opus=l_op, lam_gem=l_gm, lam_gpt=l_gp, temp=t)
                    if score > best_score:
                        best_score = score
                        best_cfg = (w_s, l_op, l_gm, l_gp, t)
                        print(f"🔥 NEW BEST: {best_score*100:.2f}% | w_svc={w_s:.1f}, opus={l_op:.1f}, gem={l_gm:.1f}, gpt={l_gp:.1f}, temp={t:.2f}")

print("\n" + "="*60)
print(f"🏆 ULTIMATE 18-CLASS BALANCED ACCURACY: {best_score*100:.2f}% with params {best_cfg}")
print("="*60, flush=True)

