import numpy as np, pandas as pd, re, time
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, diags
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

t0 = time.time()
print("🚀 [Step 1] Initializing Complete Hierarchical 4-Gate Synthesis Pipeline...", flush=True)

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

print("📚 [Step 2] Building Stylometry Views & Centered Dual-SVC OOFs...", flush=True)
_FW = set('a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little'.split())
_shape = lambda t: re.sub(r'[^\W\d_]+', lambda m_: 'X' if m_[0].isupper() else 'x', t)
_func = lambda t: ' '.join(w.lower() if w.lower() in _FW or not w.isalnum() else 'CONTENT' for w in re.findall(r'\w+|[^\w\s]', t))

matrices = []
for view, lo, hi, mf, wv in [
    ('char', 3, 5, 30000, 1.0),
    ('char_wb', 2, 4, 20000, 0.8),
    ('word', 1, 3, 15000, 0.7),
    ('shape', 2, 6, 8000, 0.5),
    ('function', 1, 3, 8000, 0.5)
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

base_tot = log_dl + 1.5 * oof_svc_plain + 1.5 * oof_svc_centered

print("🔬 [Step 3] Constructing 4 Specialized Expert Gates...", flush=True)

# GATE 1: Claude Opus Gate (3-way clean external transfer)
claude_ext = pd.read_csv('data/ntl_extra_passages_claude.csv')
opus_models = ['claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5']
sub_ext_opus = claude_ext[claude_ext['model'].isin(opus_models)].copy().reset_index(drop=True)
y_opus_train = np.array([opus_models.index(m) for m in sub_ext_opus['model'].values])

v_op1 = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=30000, sublinear_tf=True, dtype=np.float32)
v_op2 = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', max_features=20000, sublinear_tf=True, dtype=np.float32)
X_op_tr = hstack([v_op1.fit_transform(sub_ext_opus['text'].values), v_op2.fit_transform(sub_ext_opus['text'].values)], format='csr')
X_op_va = hstack([v_op1.transform(texts), v_op2.transform(texts)], format='csr')

clf_opus = LinearSVC(C=0.5, random_state=42, max_iter=2000, dual=True)
clf_opus.fit(X_op_tr, y_opus_train)
logits_opus = clf_opus.decision_function(X_op_va)
opus_idx = [L2I[m] for m in opus_models]
sc_opus = (logits_opus - logits_opus.mean(axis=1, keepdims=True)) / np.maximum(logits_opus.std(axis=1, keepdims=True), 0.25)

# GATE 2: Chinese Cluster Gate (GLM-5.3, GLM-5.3-Flash, Qwen3.8-Max, Kimi-K3)
chinese_models = ['glm-5.3', 'glm-5.3-flash', 'qwen3.8-max', 'kimi-k3']
chinese_idx = [L2I[m] for m in chinese_models]

v_cn = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=25000, sublinear_tf=True, dtype=np.float32)
X_cn = v_cn.fit_transform(texts)
oof_cn_logits = np.zeros((len(train), 4), dtype=np.float32)

for fold, (tr_pids_idx, val_pids_idx) in enumerate(kf.split(unique_pids)):
    tr_pids = unique_pids[tr_pids_idx]
    val_pids = unique_pids[val_pids_idx]
    tr_idx = np.where(np.isin(all_pids, tr_pids))[0]
    val_idx = np.where(np.isin(all_pids, val_pids))[0]
    
    tr_sub = [i for i in tr_idx if y_true[i] in chinese_idx]
    y_sub = np.array([chinese_models.index(LABELS[y_true[i]]) for i in tr_sub])
    
    clf_cn = LinearSVC(C=0.5, random_state=42, max_iter=2000, dual=True)
    clf_cn.fit(X_cn[tr_sub], y_sub)
    oof_cn_logits[val_idx] = clf_cn.decision_function(X_cn[val_idx])

sc_cn = (oof_cn_logits - oof_cn_logits.mean(axis=1, keepdims=True)) / np.maximum(oof_cn_logits.std(axis=1, keepdims=True), 0.25)

# GATE 3 & 4: Ordinal Rank-Order Features for Gemini & GPT
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
    rank_mat[idx] = p_ranks - 0.5

oof_gem_diff = np.zeros(len(train), dtype=np.float32)
oof_gpt_diff = np.zeros(len(train), dtype=np.float32)

idx_g37 = L2I['gemini-3.7-flash']
idx_g38 = L2I['gemini-3.8-flash']
idx_sol = L2I['gpt-5.6-sol']
idx_terra = L2I['gpt-5.6-terra']

v_gem_wb = TfidfVectorizer(ngram_range=(4, 6), analyzer='char_wb', max_features=12000, sublinear_tf=True, dtype=np.float32)
X_gem_wb = v_gem_wb.fit_transform(texts)
X_gem_feat = hstack([X_gem_wb, csr_matrix(rank_mat * 2.0)], format='csr')

v_gpt_w = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', max_features=10000, sublinear_tf=True, dtype=np.float32)
X_gpt_w = v_gpt_w.fit_transform(texts)
X_gpt_feat = hstack([X_gpt_w, csr_matrix(rank_mat * 2.0)], format='csr')

for fold, (tr_pids_idx, val_pids_idx) in enumerate(kf.split(unique_pids)):
    tr_pids = unique_pids[tr_pids_idx]
    val_pids = unique_pids[val_pids_idx]
    tr_idx = np.where(np.isin(all_pids, tr_pids))[0]
    val_idx = np.where(np.isin(all_pids, val_pids))[0]
    
    # Gemini
    tr_gem = [i for i in tr_idx if y_true[i] in [idx_g37, idx_g38]]
    y_gem = (y_true[tr_gem] == idx_g38).astype(int)
    clf_g = LogisticRegression(C=1.0, max_iter=500)
    clf_g.fit(X_gem_feat[tr_gem], y_gem)
    oof_gem_diff[val_idx] = clf_g.decision_function(X_gem_feat[val_idx])
    
    # GPT
    tr_gpt = [i for i in tr_idx if y_true[i] in [idx_sol, idx_terra]]
    y_gpt = (y_true[tr_gpt] == idx_terra).astype(int)
    clf_gp = LogisticRegression(C=1.0, max_iter=500)
    clf_gp.fit(X_gpt_feat[tr_gpt], y_gpt)
    oof_gpt_diff[val_idx] = clf_gp.decision_function(X_gpt_feat[val_idx])

print("✅ All 4 Expert Gates Built Successfully!", flush=True)

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

print(f"\n👉 Baseline Dual-SVC (Leak-Free 5-Fold CV): {eval_matching(base_tot)*100:.2f}%")

best_score = 0
best_params = None

for w_op in [0.25, 0.35, 0.45]:
    for w_cn in [0.10, 0.20, 0.30]:
        for w_ge in [0.05, 0.10, 0.15]:
            for w_gp in [0.05, 0.10, 0.15]:
                for tmp in [0.30, 0.35, 0.40]:
                    tot = base_tot.copy()
                    if w_op > 0:
                        tot[:, opus_idx] += w_op * sc_opus
                    if w_cn > 0:
                        tot[:, chinese_idx] += w_cn * sc_cn
                    if w_ge > 0:
                        tot[:, idx_g38] += w_ge * oof_gem_diff
                        tot[:, idx_g37] -= w_ge * oof_gem_diff
                    if w_gp > 0:
                        tot[:, idx_terra] += w_gp * oof_gpt_diff
                        tot[:, idx_sol] -= w_gp * oof_gpt_diff
                        
                    acc = eval_matching(tot, temp=tmp)
                    if acc > best_score:
                        best_score = acc
                        best_params = (w_op, w_cn, w_ge, w_gp, tmp)
                        print(f"🔥 BEST: {best_score*100:.2f}% | opus={w_op}, cn={w_cn}, gem={w_ge}, gpt={w_gp}, temp={tmp}", flush=True)

print("\n" + "="*60)
print(f"🏆 HIERARCHICAL 4-GATE COMPOSITE SCORE: {best_score*100:.2f}% with params {best_params}")
print(f"⏱️ Total Execution Time: {time.time() - t0:.2f}s")
print("="*60, flush=True)
