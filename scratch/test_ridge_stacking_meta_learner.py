import numpy as np, pandas as pd, re, time
from sklearn.linear_model import RidgeClassifier, Ridge
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

t0 = time.time()
print("🚀 [Step 1] Loading Precomputed Leak-Free Representations...", flush=True)

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
all_pids = train['prompt_id'].values
unique_pids = np.unique(all_pids)

# One-hot true targets (2699, 18)
Y_onehot = np.zeros((len(train), 18), dtype=np.float32)
Y_onehot[np.arange(len(train)), y_true] = 1.0

# Load OOF SVCs if cached or evaluate
print("⚖️ [Step 2] Training Leak-Free Multi-Task Ridge Meta-Learner...", flush=True)

# Build feature stack for stacking
# Feature 1: DL probs (18)
# Feature 2: DL log-probs (18)
log_dl = np.log(dl_blend + 1e-12)

# Test Ridge on combinations of alpha
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def eval_meta_learner(alpha=10.0, temp=0.35):
    oof_meta_logits = np.zeros((len(train), 18), dtype=np.float32)
    
    # Simple feature vector per sample
    X_meta = np.column_stack([dl_blend, log_dl])
    
    for fold, (tr_pids_idx, val_pids_idx) in enumerate(kf.split(unique_pids)):
        tr_pids = unique_pids[tr_pids_idx]
        val_pids = unique_pids[val_pids_idx]
        tr_idx = np.where(np.isin(all_pids, tr_pids))[0]
        val_idx = np.where(np.isin(all_pids, val_pids))[0]
        
        ridge = Ridge(alpha=alpha, random_state=42)
        ridge.fit(X_meta[tr_idx], Y_onehot[tr_idx])
        oof_meta_logits[val_idx] = ridge.predict(X_meta[val_idx])
        
    preds = np.zeros(len(train), dtype=int)
    for p in unique_pids:
        idx = np.where(all_pids == p)[0]
        M = oof_meta_logits[idx]
        K = np.exp(M / temp)
        for _ in range(15):
            K = K / (K.sum(axis=1, keepdims=True) + 1e-12)
            K = K / (K.sum(axis=0, keepdims=True) + 1e-12)
        r, c = linear_sum_assignment(-K)
        preds[idx[r]] = c
    return balanced_accuracy_score(y_true, preds)

for a in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    for t in [0.25, 0.35, 0.50]:
        sc = eval_meta_learner(alpha=a, temp=t)
        print(f"Alpha={a:7.1f}, Temp={t:4.2f} -> Balanced Accuracy: {sc*100:.2f}%")

