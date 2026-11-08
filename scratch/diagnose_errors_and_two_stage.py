import numpy as np, pandas as pd, re, time
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, diags
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
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

print("📚 [Step 2] Building Multi-View TF-IDF and Sparse Group Centering...", flush=True)
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

print("🔬 [Step 3] Computing Prompt-Relative Z-Scores for Dense Forensic Features...", flush=True)

wlens = np.array([len(t.split()) for t in texts], dtype=np.float32)
puncs = [':', ';', '—', '–', '-', '“', '”', '’', '‘', '(', ')', '/', '!', '?', '`', '*', '"', "'"]
raw_feats = [wlens]
for p in puncs:
    c = np.array([t.count(p) for t in texts], dtype=np.float32)
    raw_feats.append(c)
    raw_feats.append(c / (wlens + 1.0))
caps = np.array([len(re.findall(r'[A-Z]', t)) for t in texts], dtype=np.float32)
raw_feats.append(caps / (wlens + 1.0))
sents = np.array([len(re.findall(r'[.!?]+', t)) for t in texts], dtype=np.float32)
raw_feats.append(sents)
raw_feats.append(wlens / (sents + 1.0))

F_raw = np.column_stack(raw_feats)

F_prompt_z = np.zeros_like(F_raw)
for p in unique_pids:
    idx = np.where(all_pids == p)[0]
    sub = F_raw[idx]
    m = sub.mean(axis=0)
    s = np.maximum(sub.std(axis=0), 1e-4)
    F_prompt_z[idx] = (sub - m) / s

print("⚖️ [Step 4] Precomputing 5-Fold Dual-SVC and Prompt-Relative Micro-Gates...", flush=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_svc_plain = np.zeros((len(train), 18), dtype=np.float32)
oof_svc_centered = np.zeros((len(train), 18), dtype=np.float32)

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
    
    tr_gem = [i for i in tr_idx if y_true[i] in [idx_g37, idx_g38]]
    y_gem = (y_true[tr_gem] == idx_g38).astype(int)
    clf_g = LogisticRegression(C=0.5, max_iter=500)
    clf_g.fit(F_prompt_z[tr_gem], y_gem)
    oof_gemini_diff[val_idx] = clf_g.decision_function(F_prompt_z[val_idx])
    
    tr_gpt = [i for i in tr_idx if y_true[i] in [idx_sol, idx_terra]]
    y_gpt = (y_true[tr_gpt] == idx_terra).astype(int)
    clf_gp = LogisticRegression(C=0.5, max_iter=500)
    clf_gp.fit(F_prompt_z[tr_gpt], y_gpt)
    oof_gpt_diff[val_idx] = clf_gp.decision_function(F_prompt_z[val_idx])
    
    tr_glm = [i for i in tr_idx if y_true[i] in [idx_glm, idx_glm_f]]
    y_glm = (y_true[tr_glm] == idx_glm_f).astype(int)
    clf_gl = LogisticRegression(C=0.5, max_iter=500)
    clf_gl.fit(F_prompt_z[tr_glm], y_glm)
    oof_glm_diff[val_idx] = clf_gl.decision_function(F_prompt_z[val_idx])

# Train Opus Gate
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

base_tot = log_dl + 1.5 * oof_svc_plain + 1.5 * oof_svc_centered
tot = base_tot.copy()
tot[:, opus_idx] += 0.3 * sc_opus
tot[:, idx_g38] += 0.1 * oof_gemini_diff
tot[:, idx_g37] -= 0.1 * oof_gemini_diff
tot[:, idx_glm_f] += 0.2 * oof_glm_diff
tot[:, idx_glm] -= 0.2 * oof_glm_diff

print("📊 [Step 5] Analyzing Per-Class Balanced Accuracy & Confusion...", flush=True)

temp = 0.35
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

cm = confusion_matrix(y_true, preds)
print("\n--- Per-Class Accuracy ---")
for i, l in enumerate(LABELS):
    class_acc = cm[i, i] / cm[i].sum()
    print(f"{l:25s}: {class_acc*100:5.2f}% ({cm[i, i]}/{cm[i].sum()})")

print(f"\nOverall Balanced Accuracy: {balanced_accuracy_score(y_true, preds)*100:.2f}%")

print("\n--- Top Confused Pairs ---")
confusions = []
for i in range(18):
    for j in range(18):
        if i != j and cm[i, j] > 5:
            confusions.append((cm[i, j], LABELS[i], LABELS[j]))
confusions.sort(reverse=True)
for count, l1, l2 in confusions[:15]:
    print(f"True: {l1:22s} -> Pred: {l2:22s} ({count} errors)")

print("\n🔬 [Step 6] Testing Two-Stage Anchor-Locked Assignment...", flush=True)

# Define Family Clusters
FAMILY_CLUSTERS = [
    [L2I['claude-opus-4-6'], L2I['claude-opus-4-8'], L2I['claude-opus-5']],
    [L2I['gemini-3.7-flash'], L2I['gemini-3.8-flash']],
    [L2I['gpt-5.6-sol'], L2I['gpt-5.6-terra']],
    [L2I['glm-5.3'], L2I['glm-5.3-flash']],
]
all_cluster_members = set([idx for cluster in FAMILY_CLUSTERS for idx in cluster])
anchor_members = [i for i in range(18) if i not in all_cluster_members]
print(f"Anchors ({len(anchor_members)}): {[LABELS[i] for i in anchor_members]}")

def eval_two_stage(logits, temp=0.35, anchor_threshold=0.0):
    two_stage_preds = np.zeros(len(train), dtype=int)
    for p in unique_pids:
        idx = np.where(all_pids == p)[0]
        n_passages = len(idx)
        M = logits[idx] # (n_passages, 18)
        
        # Method A: Hungarian on full 18x18
        K = np.exp(M / temp)
        for _ in range(15):
            K = K / (K.sum(axis=1, keepdims=True) + 1e-12)
            K = K / (K.sum(axis=0, keepdims=True) + 1e-12)
            
        r_all, c_all = linear_sum_assignment(-K)
        initial_assignment = {r: c for r, c in zip(r_all, c_all)}
        
        # For each family cluster, isolate the passages assigned to that cluster
        # and re-solve sub-Hungarian exclusively within the cluster
        final_assignment = dict(initial_assignment)
        
        for cluster in FAMILY_CLUSTERS:
            # find which passages were assigned to any model in this cluster
            cluster_passages = [r for r, c in initial_assignment.items() if c in cluster]
            if len(cluster_passages) == len(cluster):
                # sub-matrix
                sub_M = M[np.ix_(cluster_passages, cluster)]
                sub_r, sub_c = linear_sum_assignment(-sub_M)
                for pr, pc in zip(sub_r, sub_c):
                    final_assignment[cluster_passages[pr]] = cluster[pc]
                    
        for r, c in final_assignment.items():
            two_stage_preds[idx[r]] = c
            
    return balanced_accuracy_score(y_true, two_stage_preds)

score_two_stage = eval_two_stage(tot, temp=0.35)
print(f"👉 Two-Stage Sub-Cluster Re-assignment Score: {score_two_stage*100:.2f}%")

