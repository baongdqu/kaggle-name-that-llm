import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
import re, time, sys

t0 = time.time()
print("🚀 [Step 1] Loading Datasets and Deep Learning OOFs...", flush=True)

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

dl1 = d1['probs'][:, idx_map1]
dl2 = d2['probs'][:, idx_map2]
dl3 = d3['probs'][:, idx_map3]

# Optimal DL blend
dl_blend = (dl1 + dl2 + dl3) / 3.0
log_dl = np.log(dl_blend + 1e-12)

all_pids = train['prompt_id'].values
unique_pids = np.unique(all_pids)

def run_hungarian(log_mat):
    preds = np.zeros(len(train), dtype=int)
    for pid in unique_pids:
        idx = np.where(all_pids == pid)[0]
        cost = -log_mat[idx]
        r, c = linear_sum_assignment(cost)
        preds[idx[r]] = c
    return balanced_accuracy_score(y_true, preds)

base_dl_acc = run_hungarian(log_dl)
print(f"✅ Deep Learning OOF + Hungarian Baseline: {base_dl_acc*100:.2f}%\n", flush=True)

print("🛠️ [Step 2] Engineering Multi-View Stylometric Features...", flush=True)
texts = train['text'].astype(str).tolist()
_FW = set('a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little'.split())
_shape = lambda t: re.sub(r'[^\W\d_]+', lambda m_: 'X' if m_[0].isupper() else 'x', t)
_func = lambda t: ' '.join(w.lower() if w.lower() in _FW or not w.isalnum() else 'CONTENT' for w in re.findall(r'\w+|[^\w\s]', t))

matrices = []
for view, lo, hi, mf, wv in [
    ('char', 3, 5, 40000, 1.0),
    ('char_wb', 2, 4, 30000, 0.8),
    ('word', 1, 3, 25000, 0.7),
    ('shape', 2, 6, 15000, 0.5),
    ('function', 1, 3, 10000, 0.5)
]:
    prep = _shape if view == 'shape' else _func if view == 'function' else (lambda s_: s_)
    an = 'char' if view == 'shape' else 'word' if view == 'function' else ('char_wb' if view == 'char_wb' else view)
    o = dict(analyzer=an, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf, dtype=np.float32)
    if an == 'word': o['token_pattern'] = r'(?u)\b\w+\b|[^\w\s]'
    v = TfidfVectorizer(**o)
    matrices.append(v.fit_transform([prep(t) for t in texts]) * wv)

# Physical numerical features
wlens = np.array([len(t.split()) for t in texts], dtype=np.float32)
clens = np.array([len(t) for t in texts], dtype=np.float32)
em_dash = np.array([t.count('—') for t in texts], dtype=np.float32)
semi = np.array([t.count(';') for t in texts], dtype=np.float32)
colon = np.array([t.count(':') for t in texts], dtype=np.float32)
qmark = np.array([t.count('?') for t in texts], dtype=np.float32)
exclam = np.array([t.count('!') for t in texts], dtype=np.float32)
comma = np.array([t.count(',') for t in texts], dtype=np.float32)
curly_single = np.array([t.count('’') + t.count('‘') for t in texts], dtype=np.float32)
straight_single = np.array([t.count("'") for t in texts], dtype=np.float32)
curly_double = np.array([t.count('“') + t.count('”') for t in texts], dtype=np.float32)
straight_double = np.array([t.count('"') for t in texts], dtype=np.float32)
digits = np.array([sum(c.isdigit() for c in t) / max(1.0, len(t)) for t in texts], dtype=np.float32)
caps = np.array([sum(c.isupper() for c in t) / max(1.0, len(t)) for t in texts], dtype=np.float32)

# Prompt-relative features
rel_wlens = np.zeros(len(train), dtype=np.float32)
rel_clens = np.zeros(len(train), dtype=np.float32)
for pid in unique_pids:
    idx = np.where(all_pids == pid)[0]
    sw = wlens[idx]
    sc = clens[idx]
    rel_wlens[idx] = (sw - sw.mean()) / max(1.0, sw.std())
    rel_clens[idx] = (sc - sc.mean()) / max(1.0, sc.std())

dense_feats = np.column_stack([
    rel_wlens, rel_clens, em_dash, semi, colon, qmark, exclam, comma, 
    curly_single, straight_single, curly_double, straight_double, digits, caps,
    wlens / 100.0, clens / 500.0
])
matrices.append(csr_matrix(dense_feats * 2.0, dtype=np.float32))

X_all = hstack(matrices, format='csr')
print(f"✅ Total Extracted Feature Matrix: {X_all.shape[1]} features in {time.time()-t0:.1f}s\n", flush=True)

print("🧠 [Step 3] Precomputing Fast 5-Fold OOF Predictions (Dual LibLinear)...", flush=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_svc1 = np.zeros((len(train), 18), dtype=np.float32)
oof_svc2 = np.zeros((len(train), 18), dtype=np.float32)
oof_ridge = np.zeros((len(train), 18), dtype=np.float32)

MICRO_CLUSTERS = {
    'gemini_flash': ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'],
    'gpt56_trio': ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra'],
    'glm_fable': ['glm-5.3', 'glm-5.3-flash', 'claude-fable-5'],
    'opus_trio': ['claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5']
}

oof_clusters = {cname: np.zeros((len(train), len(cmodels)), dtype=np.float32) 
                for cname, cmodels in MICRO_CLUSTERS.items()}

fold_time = time.time()
for fold, (tr_pids_idx, val_pids_idx) in enumerate(kf.split(unique_pids)):
    tr_pids = unique_pids[tr_pids_idx]
    val_pids = unique_pids[val_pids_idx]
    tr_idx = np.where(np.isin(all_pids, tr_pids))[0]
    val_idx = np.where(np.isin(all_pids, val_pids))[0]
    
    # 1. LinearSVC C=0.3
    svc1 = LinearSVC(C=0.3, random_state=42, max_iter=2000)
    svc1.fit(X_all[tr_idx], y_true[tr_idx])
    d1 = svc1.decision_function(X_all[val_idx])
    d1_m = d1.max(axis=1, keepdims=True)
    exp_d1 = np.exp(d1 - d1_m)
    oof_svc1[val_idx] = np.log(exp_d1 / exp_d1.sum(axis=1, keepdims=True) + 1e-12)
    
    # 2. LinearSVC C=1.0
    svc2 = LinearSVC(C=1.0, random_state=42, max_iter=2000)
    svc2.fit(X_all[tr_idx], y_true[tr_idx])
    d2 = svc2.decision_function(X_all[val_idx])
    d2_m = d2.max(axis=1, keepdims=True)
    exp_d2 = np.exp(d2 - d2_m)
    oof_svc2[val_idx] = np.log(exp_d2 / exp_d2.sum(axis=1, keepdims=True) + 1e-12)
    
    # 3. RidgeClassifier alpha=1.0
    ridge = RidgeClassifier(alpha=1.0, random_state=42)
    ridge.fit(X_all[tr_idx], y_true[tr_idx])
    dr = ridge.decision_function(X_all[val_idx])
    dr_m = dr.max(axis=1, keepdims=True)
    exp_dr = np.exp(dr - dr_m)
    oof_ridge[val_idx] = np.log(exp_dr / exp_dr.sum(axis=1, keepdims=True) + 1e-12)
    
    # 4. Micro-Cluster Specialists
    for cname, cmodels in MICRO_CLUSTERS.items():
        c_indices = [L2I[m] for m in cmodels]
        sub_tr = [i for i in tr_idx if y_true[i] in c_indices]
        sub_y = [c_indices.index(y_true[i]) for i in sub_tr]
        
        c_clf = LinearSVC(C=1.0, max_iter=2000, random_state=42)
        c_clf.fit(X_all[sub_tr], sub_y)
        c_dec = c_clf.decision_function(X_all[val_idx])
        c_dec_m = c_dec.max(axis=1, keepdims=True)
        exp_c = np.exp(c_dec - c_dec_m)
        oof_clusters[cname][val_idx] = np.log(exp_c / exp_c.sum(axis=1, keepdims=True) + 1e-12)
        
    print(f"  Fold {fold+1}/5 trained in {time.time()-fold_time:.1f}s", flush=True)
    fold_time = time.time()

print(f"✅ All 5 folds precomputed! Starting hyperparameter optimization...\n", flush=True)

# Grid Search over weights
best_acc = 0.0
best_cfg = None

results = []
for w_s1 in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
    for w_s2 in [0.0, 0.5, 1.0, 1.5, 2.0]:
        for w_r in [0.0, 0.5, 1.0]:
            for w_c in [0.0, 0.5, 1.0, 1.5, 2.0]:
                tot_log = log_dl + w_s1 * oof_svc1 + w_s2 * oof_svc2 + w_r * oof_ridge
                
                if w_c > 0:
                    tot_log = tot_log.copy()
                    for cname, cmodels in MICRO_CLUSTERS.items():
                        c_indices = [L2I[m] for m in cmodels]
                        for k, glob_cls in enumerate(c_indices):
                            tot_log[:, glob_cls] += w_c * oof_clusters[cname][:, k]
                
                acc = run_hungarian(tot_log)
                results.append((acc, w_s1, w_s2, w_r, w_c))
                if acc > best_acc:
                    best_acc = acc
                    best_cfg = (w_s1, w_s2, w_r, w_c)
                    print(f"🔥 NEW BEST: {best_acc*100:.2f}% | w_s1={w_s1} | w_s2={w_s2} | w_r={w_r} | w_cluster={w_c}", flush=True)

results.sort(key=lambda x: x[0], reverse=True)
print("\n" + "="*50)
print(f"🏆 Top 5 Optimal Configurations:")
for i, (acc, ws1, ws2, wr, wc) in enumerate(results[:5]):
    print(f"  #{i+1}: Acc = {acc*100:.2f}% | SVC_0.3 = {ws1} | SVC_1.0 = {ws2} | Ridge = {wr} | Cluster = {wc}")
print("="*50, flush=True)
