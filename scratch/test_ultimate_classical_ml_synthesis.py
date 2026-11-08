import numpy as np, pandas as pd, re, time
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, diags
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

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
texts = train['text'].astype(str).tolist()

print("📚 [Step 2] Building 5-View Stylometry & Sparse Group Centering...", flush=True)
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

print("⚖️ [Step 3] Precomputing Leak-Free 5-Fold Dual-SVC Base OOFs...", flush=True)
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

base_tot = log_dl + 1.5 * oof_svc_plain + 1.5 * oof_svc_centered

print("🔬 [Step 4] Precomputing Family-Adaptive Sibling Micro-Gates...", flush=True)

# 1. Clean Opus Gate (71.11% transfer)
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
logits_opus = clf_opus.decision_function(X_op_va)
opus_idx = [L2I[m] for m in opus_models]

sc_opus = logits_opus - logits_opus.mean(axis=1, keepdims=True)
sc_opus = sc_opus / np.maximum(sc_opus.std(axis=1, keepdims=True), 0.25)

# 2. Prompt-Relative Z-Scores for Gemini, GPT, GLM
wlens = np.array([len(t.split()) for t in texts], dtype=np.float32)
punc_chars = [':', ';', '—', '–', '-', '“', '”', '’', '‘', '(', ')', '/', '!', '?', '`', '*', '"', "'"]

z_features = []
# Word length z-score
z_w = np.zeros_like(wlens)
for p in unique_pids:
    idx = np.where(all_pids == p)[0]
    m_ = wlens[idx].mean()
    s_ = max(1.0, wlens[idx].std())
    z_w[idx] = (wlens[idx] - m_) / s_
z_features.append(z_w)

for punc in punc_chars:
    cnts = np.array([t.count(punc) for t in texts], dtype=np.float32)
    z_c = np.zeros_like(cnts)
    for p in unique_pids:
        idx = np.where(all_pids == p)[0]
        m_ = cnts[idx].mean()
        s_ = max(1.0, cnts[idx].std())
        z_c[idx] = (cnts[idx] - m_) / s_
    z_features.append(z_c)
    z_features.append((cnts > 0).astype(np.float32))

F_dense_z = np.column_stack(z_features)

oof_gemini_diff = np.zeros(len(train), dtype=np.float32)
oof_gpt_diff = np.zeros(len(train), dtype=np.float32)
oof_glm_diff = np.zeros(len(train), dtype=np.float32)

idx_g37 = L2I['gemini-3.7-flash']
idx_g38 = L2I['gemini-3.8-flash']
idx_sol = L2I['gpt-5.6-sol']
idx_terra = L2I['gpt-5.6-terra']
idx_glm = L2I['glm-5.3']
idx_glm_f = L2I['glm-5.3-flash']

for fold, (tr_pids_idx, val_pids_idx) in enumerate(kf.split(unique_pids)):
    tr_pids = unique_pids[tr_pids_idx]
    val_pids = unique_pids[val_pids_idx]
    tr_idx = np.where(np.isin(all_pids, tr_pids))[0]
    val_idx = np.where(np.isin(all_pids, val_pids))[0]
    
    # Gemini 3.7 vs 3.8
    tr_gem = [i for i in tr_idx if y_true[i] in [idx_g37, idx_g38]]
    y_gem = (y_true[tr_gem] == idx_g38).astype(int)
    clf_g = LogisticRegression(C=1.0, max_iter=500)
    clf_g.fit(F_dense_z[tr_gem], y_gem)
    oof_gemini_diff[val_idx] = clf_g.decision_function(F_dense_z[val_idx])
    
    # GPT Sol vs Terra
    tr_gpt = [i for i in tr_idx if y_true[i] in [idx_sol, idx_terra]]
    y_gpt = (y_true[tr_gpt] == idx_terra).astype(int)
    clf_gp = LogisticRegression(C=1.0, max_iter=500)
    clf_gp.fit(F_dense_z[tr_gpt], y_gpt)
    oof_gpt_diff[val_idx] = clf_gp.decision_function(F_dense_z[val_idx])
    
    # GLM 5.3 vs Flash
    tr_glm = [i for i in tr_idx if y_true[i] in [idx_glm, idx_glm_f]]
    y_glm = (y_true[tr_glm] == idx_glm_f).astype(int)
    clf_gl = LogisticRegression(C=1.0, max_iter=500)
    clf_gl.fit(F_dense_z[tr_glm], y_glm)
    oof_glm_diff[val_idx] = clf_gl.decision_function(F_dense_z[val_idx])

print("✅ Family-Adaptive Micro-Gates ready!", flush=True)

def eval_matching(logits, temp=0.35):
    preds = np.zeros(len(train), dtype=int)
    for p in unique_pids:
        idx = np.where(all_pids == p)[0]
        M = logits[idx]
        K = np.exp(M / temp)
        for _ in range(15):
            K = K / (K.sum(axis=1, keepdims=True) + 1e-12)
            K = K / (K.sum(axis=0, keepdims=True) + 1e-12)
        r, c = linear_sum_assignment(-K)
        preds[idx[r]] = c
    return balanced_accuracy_score(y_true, preds)

print(f"\n👉 Baseline Dual-SVC (5 Views + DL): {eval_matching(base_tot)*100:.2f}%")

best_score = 0
best_params = None

for w_op in [0.2, 0.3, 0.4]:
    for w_ge in [0.0, 0.05, 0.1]:
        for w_gp in [0.05, 0.1, 0.15]:
            for w_gl in [0.05, 0.1, 0.15]:
                for tmp in [0.30, 0.35, 0.40]:
                    tot = base_tot.copy()
                    if w_op > 0:
                        tot[:, opus_idx] += w_op * sc_opus
                    if w_ge > 0:
                        tot[:, idx_g38] += w_ge * oof_gemini_diff
                        tot[:, idx_g37] -= w_ge * oof_gemini_diff
                    if w_gp > 0:
                        tot[:, idx_terra] += w_gp * oof_gpt_diff
                        tot[:, idx_sol] -= w_gp * oof_gpt_diff
                    if w_gl > 0:
                        tot[:, idx_glm_f] += w_gl * oof_glm_diff
                        tot[:, idx_glm] -= w_gl * oof_glm_diff
                        
                    acc = eval_matching(tot, temp=tmp)
                    if acc > best_score:
                        best_score = acc
                        best_params = (w_op, w_ge, w_gp, w_gl, tmp)
                        print(f"🔥 BEST: {best_score*100:.2f}% | opus={w_op}, gem={w_ge}, gpt={w_gp}, glm={w_gl}, temp={tmp}", flush=True)

print("\n" + "="*60)
print(f"🏆 ULTIMATE SYNTHESIS SCORE: {best_score*100:.2f}% with params {best_params}")
print("="*60, flush=True)
