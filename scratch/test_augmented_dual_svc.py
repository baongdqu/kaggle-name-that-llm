import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, diags, vstack
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

print("📚 [Step 2] Curating Clean Balanced External Pool...", flush=True)
# Load external short wide
df_ext = pd.read_parquet('data/external_short_wide.parquet')
ext_texts, ext_labels = [], []
for col in LABELS:
    if col in df_ext.columns:
        s = df_ext[col].dropna()
        # Cap at 800 per model for balance
        s_sampled = s.sample(min(len(s), 800), random_state=42)
        for t in s_sampled:
            ext_texts.append(str(t))
            ext_labels.append(L2I[col])

# Load Claude extra
df_claude = pd.read_csv('data/ntl_extra_passages_claude.csv')
for m in df_claude['model'].unique():
    if m in L2I:
        s = df_claude[df_claude['model'] == m]['text'].dropna()
        s_sampled = s.sample(min(len(s), 500), random_state=42)
        for t in s_sampled:
            ext_texts.append(str(t))
            ext_labels.append(L2I[m])

ext_y = np.array(ext_labels, dtype=int)
print(f"✅ Clean External Pool: {len(ext_texts)} passages across 18 models!", flush=True)

print("🛠️ [Step 3] Multi-View TF-IDF on Combined Train + External Pool...", flush=True)
all_corpus = train['text'].astype(str).tolist() + ext_texts

_FW = set('a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little'.split())
_shape = lambda t: re.sub(r'[^\W\d_]+', lambda m_: 'X' if m_[0].isupper() else 'x', t)
_func = lambda t: ' '.join(w.lower() if w.lower() in _FW or not w.isalnum() else 'CONTENT' for w in re.findall(r'\w+|[^\w\s]', t))

matrices = []
for view, lo, hi, mf, wv in [
    ('char', 3, 5, 40000, 1.0),
    ('char_wb', 2, 4, 30000, 0.8),
    ('word', 1, 3, 25000, 0.7),
    ('shape', 2, 6, 12000, 0.5),
    ('function', 1, 3, 10000, 0.5)
]:
    prep = _shape if view == 'shape' else _func if view == 'function' else (lambda s_: s_)
    an = 'char' if view == 'shape' else 'word' if view == 'function' else ('char_wb' if view == 'char_wb' else view)
    o = dict(analyzer=an, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf, dtype=np.float32)
    if an == 'word': o['token_pattern'] = r'(?u)\b\w+\b|[^\w\s]'
    v = TfidfVectorizer(**o)
    matrices.append(v.fit_transform([prep(t) for t in all_corpus]) * wv)

X_combined = hstack(matrices, format='csr')
N_train = len(train)
X_train = X_combined[:N_train]
X_ext = X_combined[N_train:]

print(f"✅ Combined Matrix: {X_combined.shape[1]} features in {time.time()-t0:.1f}s", flush=True)

print("🧠 [Step 4] Precomputing 5-Fold Leak-Free Dual-SVC with Augmented Pool...", flush=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_svc_aug = np.zeros((N_train, 18), dtype=np.float32)

for fold, (tr_pids_idx, val_pids_idx) in enumerate(kf.split(unique_pids)):
    t_f = time.time()
    tr_pids = unique_pids[tr_pids_idx]
    val_pids = unique_pids[val_pids_idx]
    tr_idx = np.where(np.isin(all_pids, tr_pids))[0]
    val_idx = np.where(np.isin(all_pids, val_pids))[0]
    
    # Train on training fold + external pool
    X_tr_aug = vstack([X_train[tr_idx], X_ext]).tocsr()
    y_tr_aug = np.concatenate([y_true[tr_idx], ext_y])
    
    # Sample weights: give official train rows 2.0x weight and external rows 0.5x
    sw = np.concatenate([np.full(len(tr_idx), 2.0, dtype=np.float32), np.full(len(ext_y), 0.5, dtype=np.float32)])
    
    svc = LinearSVC(C=0.5, random_state=42, max_iter=2000, dual=True)
    svc.fit(X_tr_aug, y_tr_aug, sample_weight=sw)
    
    d_val = svc.decision_function(X_train[val_idx])
    exp_d = np.exp(d_val - d_val.max(axis=1, keepdims=True))
    oof_svc_aug[val_idx] = np.log(exp_d / exp_d.sum(axis=1, keepdims=True) + 1e-12)
    
    gc.collect()
    print(f"  Fold {fold+1}/5 completed in {time.time()-t_f:.1f}s", flush=True)

print("\n📊 [Step 5] Evaluating Augmented SOTA Model...", flush=True)

def eval_augmented(w_svc, temp=0.5):
    preds = np.zeros(N_train, dtype=int)
    tot = log_dl + w_svc * oof_svc_aug
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

print(f"👉 Baseline DL only:                       {eval_augmented(0.0)*100:.2f}%")
for w in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    for t in [0.35, 0.5, 0.7]:
        acc = eval_augmented(w, temp=t)
        print(f"🔥 Augmented SVC (w={w:.1f}, temp={t:.2f}) -> Balanced Accuracy: {acc*100:.2f}%")
