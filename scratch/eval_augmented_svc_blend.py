import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, vstack
from sklearn.svm import LinearSVC
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
import time, gc

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
df_ext = pd.read_parquet('data/external_short_wide.parquet')
ext_texts, ext_labels = [], []
for col in LABELS:
    if col in df_ext.columns:
        s = df_ext[col].dropna()
        s_sampled = s.sample(min(len(s), 500), random_state=42)
        for t in s_sampled:
            ext_texts.append(str(t))
            ext_labels.append(L2I[col])

df_claude = pd.read_csv('data/ntl_extra_passages_claude.csv')
for m in df_claude['model'].unique():
    if m in L2I:
        s = df_claude[df_claude['model'] == m]['text'].dropna()
        s_sampled = s.sample(min(len(s), 400), random_state=42)
        for t in s_sampled:
            ext_texts.append(str(t))
            ext_labels.append(L2I[m])

ext_y = np.array(ext_labels, dtype=int)
print(f"✅ Total External Samples: {len(ext_texts)} across 18 models!", flush=True)

print("🛠️ [Step 3] Fitting Multi-View TF-IDF Vectorizers...", flush=True)
train_texts = train['text'].astype(str).tolist()
all_texts = train_texts + ext_texts

t1 = time.time()
v_char = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=35000, sublinear_tf=True, dtype=np.float32)
v_word = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', max_features=25000, sublinear_tf=True, dtype=np.float32)

X_all = hstack([v_char.fit_transform(all_texts), v_word.fit_transform(all_texts)], format='csr')
N_train = len(train)
X_train = X_all[:N_train]
X_ext = X_all[N_train:]
print(f"✅ Features ({X_all.shape[1]}) created in {time.time()-t1:.1f}s", flush=True)

print("🧠 [Step 4] Running 5-Fold Leak-Free Cross-Validation...", flush=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_svc_aug = np.zeros((N_train, 18), dtype=np.float32)

for fold, (tr_pids_idx, val_pids_idx) in enumerate(kf.split(unique_pids)):
    t_f = time.time()
    tr_pids = unique_pids[tr_pids_idx]
    val_pids = unique_pids[val_pids_idx]
    tr_idx = np.where(np.isin(all_pids, tr_pids))[0]
    val_idx = np.where(np.isin(all_pids, val_pids))[0]
    
    # Augmented training set: local fold + external pool
    X_tr_aug = vstack([X_train[tr_idx], X_ext]).tocsr()
    y_tr_aug = np.concatenate([y_true[tr_idx], ext_y])
    
    # Official train rows get weight 2.0, external pool gets 0.7
    sw = np.concatenate([np.full(len(tr_idx), 2.0, dtype=np.float32), np.full(len(ext_y), 0.7, dtype=np.float32)])
    
    svc = LinearSVC(C=0.3, tol=1e-3, random_state=42, max_iter=1000, dual=True)
    svc.fit(X_tr_aug, y_tr_aug, sample_weight=sw)
    
    d_val = svc.decision_function(X_train[val_idx])
    exp_d = np.exp(d_val - d_val.max(axis=1, keepdims=True))
    oof_svc_aug[val_idx] = np.log(exp_d / exp_d.sum(axis=1, keepdims=True) + 1e-12)
    
    gc.collect()
    print(f"  Fold {fold+1}/5 completed in {time.time()-t_f:.1f}s", flush=True)

print("\n📊 [Step 5] Grid Search & Balanced Accuracy Benchmark...", flush=True)

def eval_blend(w_svc, temp=0.5):
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

print(f"👉 Baseline Deep Learning (Q14B+Q8B+DeB):        {eval_blend(0.0)*100:.2f}%")

best_acc = 0
best_params = None
for w in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    for t in [0.35, 0.5, 0.7, 1.0]:
        acc = eval_blend(w, temp=t)
        if acc > best_acc:
            best_acc = acc
            best_params = (w, t)
            print(f"🔥 NEW BEST: {best_acc*100:.2f}% | w_svc={w:.1f}, temp={t:.2f}")

print("\n" + "="*60)
print(f"🏆 Best Augmented Pool CV: {best_acc*100:.2f}% with params {best_params}")
print("="*60, flush=True)

