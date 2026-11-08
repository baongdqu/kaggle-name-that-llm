import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
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

# Build Comprehensive Multi-View Features
texts = train['text'].astype(str).tolist()
_FW = set('a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little'.split())
_shape = lambda t: re.sub(r'[^\W\d_]+', lambda m_: 'X' if m_[0].isupper() else 'x', t)
_func = lambda t: ' '.join(w.lower() if w.lower() in _FW or not w.isalnum() else 'CONTENT' for w in re.findall(r'\w+|[^\w\s]', t))

matrices = []
for view, lo, hi, mf, wv in [('char', 3, 5, 30000, 1.0), ('word', 1, 3, 20000, 0.7), ('shape', 2, 6, 10000, 0.5), ('function', 1, 3, 10000, 0.5)]:
    prep = _shape if view == 'shape' else _func if view == 'function' else (lambda s_: s_)
    an = 'char' if view == 'shape' else 'word' if view == 'function' else view
    o = dict(analyzer=an, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf, dtype=np.float32)
    if an == 'word': o['token_pattern'] = r'(?u)\b\w+\b|[^\w\s]'
    v = TfidfVectorizer(**o)
    matrices.append(v.fit_transform([prep(t) for t in texts]) * wv)

X_all = hstack(matrices, format='csr')

# Prompt-relative length features
wlens = np.array([len(t.split()) for t in texts], dtype=np.float32)
em_dash = np.array([t.count('—') for t in texts], dtype=np.float32)
semi = np.array([t.count(';') for t in texts], dtype=np.float32)
colon = np.array([t.count(':') for t in texts], dtype=np.float32)

unique_pids = train['prompt_id'].unique()
rel_wlens = np.zeros(len(train), dtype=np.float32)
for pid in unique_pids:
    idx = np.where(train['prompt_id'] == pid)[0]
    sub_w = wlens[idx]
    std = sub_w.std()
    rel_wlens[idx] = (sub_w - sub_w.mean()) / max(1.0, std)

# The Top Confused Micro-Clusters identified by diagnostic
MICRO_CLUSTERS = [
    ('gemini_flash', ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash']),
    ('gpt56_trio', ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra']),
    ('glm_claude_pair', ['glm-5.3', 'glm-5.3-flash', 'claude-fable-5']),
    ('opus_trio', ['claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5'])
]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_pipeline(svc_weight, micro_weight, temp_hungarian=1.0):
    all_preds = np.zeros(len(train), dtype=int)
    for tr_pids_idx, val_pids_idx in kf.split(unique_pids):
        tr_pids = unique_pids[tr_pids_idx]
        val_pids = unique_pids[val_pids_idx]
        tr_idx = np.where(train['prompt_id'].isin(tr_pids))[0]
        val_idx = np.where(train['prompt_id'].isin(val_pids))[0]
        
        # Train global LinearSVC
        svc = LinearSVC(C=0.5, random_state=42, max_iter=2000)
        svc.fit(X_all[tr_idx], y_true[tr_idx])
        d_val = svc.decision_function(X_all[val_idx])
        log_ml_val = d_val - d_val.max(1, keepdims=True) - np.log(np.exp(d_val - d_val.max(1, keepdims=True)).sum(1, keepdims=True))
        
        tot_val = (log_dl[val_idx] + svc_weight * log_ml_val).copy()
        
        # Train and apply Micro-Cluster Specialists
        for cname, cmodels in MICRO_CLUSTERS:
            c_indices = [L2I[m] for m in cmodels]
            sub_tr = [i for i in tr_idx if y_true[i] in c_indices]
            sub_y = [c_indices.index(y_true[i]) for i in sub_tr]
            
            # Specialist Logistic Regression
            clf = LogisticRegression(C=2.0, max_iter=1000)
            clf.fit(X_all[sub_tr], sub_y)
            
            # Predict log-probabilities on val
            val_logprobs = clf.predict_log_proba(X_all[val_idx])
            
            # Blend into corresponding columns of tot_val
            for local_k, glob_cls in enumerate(c_indices):
                tot_val[:, glob_cls] += micro_weight * val_logprobs[:, local_k]
                
        # Hungarian Bipartite Assignment per Prompt
        for pid in val_pids:
            p_local_idx = np.where(train['prompt_id'].iloc[val_idx].values == pid)[0]
            cost = -tot_val[p_local_idx] / temp_hungarian
            r, c = linear_sum_assignment(cost)
            all_preds[val_idx[p_local_idx[r]]] = c
            
    acc = balanced_accuracy_score(y_true, all_preds)
    return acc

print(f"Features prepared in {time.time()-t0:.1f}s. Starting Grid Search...")
best_acc, best_params = 0.0, None

for sw in [1.5, 2.0, 2.5, 3.0, 4.0]:
    for mw in [0.5, 1.0, 1.5, 2.0, 3.0]:
        acc = evaluate_pipeline(sw, mw, temp_hungarian=1.0)
        print(f"svc_w={sw:3.1f} | micro_w={mw:3.1f} -> Balanced Accuracy: {acc*100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_params = (sw, mw)

print(f"\n🏆 Best Local Leak-Free Balanced Accuracy: {best_acc*100:.2f}% with params {best_params}")

