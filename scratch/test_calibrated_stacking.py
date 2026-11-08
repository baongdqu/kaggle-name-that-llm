import numpy as np, pandas as pd, time
from sklearn.linear_model import RidgeClassifier, LogisticRegression, Ridge
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

t0 = time.time()
print("🚀 Testing Calibrated Multi-Task Stacking...", flush=True)

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

all_pids = train['prompt_id'].values
unique_pids = np.unique(all_pids)

# Let's stack the 3 DL model probability matrices
# Features per sample: 18 x 3 = 54 probabilities
X_meta = np.column_stack([m1, m2, m3])

# Target one-hot (2699, 18)
Y_onehot = np.zeros((len(train), 18), dtype=np.float32)
Y_onehot[np.arange(len(train)), y_true] = 1.0

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_stacked_logits = np.zeros((len(train), 18), dtype=np.float32)

for alpha in [1.0, 10.0, 50.0, 100.0]:
    for tr_p_idx, val_p_idx in kf.split(unique_pids):
        tr_p = unique_pids[tr_p_idx]
        val_p = unique_pids[val_p_idx]
        tr_idx = np.where(np.isin(all_pids, tr_p))[0]
        val_idx = np.where(np.isin(all_pids, val_p))[0]
        
        # Multi-output Ridge regression
        ridge = Ridge(alpha=alpha, random_state=42)
        ridge.fit(X_meta[tr_idx], Y_onehot[tr_idx])
        oof_stacked_logits[val_idx] = ridge.predict(X_meta[val_idx])
        
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

    score = eval_matching(oof_stacked_logits)
    print(f"👉 Multi-Output Ridge Meta-Learner (alpha={alpha}): {score*100:.2f}%")

# Compare with simple scalar blend
dl_blend = (1.0 * m1 + 1.0 * m2 + 1.5 * m3) / 3.5
log_dl = np.log(dl_blend + 1e-12)
print(f"👉 Simple Weighted Blend Baseline:              {eval_matching(log_dl)*100:.2f}%")
