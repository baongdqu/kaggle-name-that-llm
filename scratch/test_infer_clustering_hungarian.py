import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score

train = pd.read_csv('data/train.csv')
pids = train['prompt_id'].values
y_true = train['model'].map(lambda x: sorted(train['model'].unique()).index(x)).values

def capped_agglo(S, cap=18, thr=0.02):
    n = len(S); A = S.astype(np.float32).copy(); np.fill_diagonal(A, -np.inf)
    size = np.ones(n, np.int32); alive = np.ones(n, bool); lab = np.arange(n)
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        M = np.where(ok, A, -np.inf); k = int(np.argmax(M)); i, j = divmod(k, n)
        if not np.isfinite(M[i, j]) or M[i, j] < thr: break
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new; A[:, i] = new; A[i, i] = -np.inf; A[j] = -np.inf; A[:, j] = -np.inf
        size[i] += size[j]; alive[j] = False; lab[lab == j] = i
    return lab

print("Running capped_agglo prompt recovery on train.csv...", flush=True)
v = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05)
Xc = v.fit_transform(train['text'].values)
sim = (Xc @ Xc.T).toarray()

# Cluster into capped groups of max size 18
Gte = capped_agglo(sim, cap=18, thr=0.02)
unique_clusters, counts = np.unique(Gte, return_counts=True)
print(f"Discovered {len(unique_clusters)} clusters! Sizes: min={counts.min()}, max={counts.max()}, median={np.median(counts):.1f}")
print(f"Clusters of size >= 10: {(counts >= 10).sum()} / {len(unique_clusters)}")

# Load champion logits
data = np.load('scratch/cache_cv_data.npz', allow_pickle=True)
base_logits = data['base_logits']

# 1. Independent argmax (what was submitted in V92 with --perpassage 1)
pred_argmax = base_logits.argmax(axis=1)
acc_argmax = balanced_accuracy_score(y_true, pred_argmax)
print(f"\n1. V92 Mode (--perpassage 1, Independent Argmax): {acc_argmax*100:.2f}% (LB: 0.664)")

# 2. Hungarian with recovered clusters (capped_agglo)
pred_hung = pred_argmax.copy()
for g in unique_clusters:
    idx = np.where(Gte == g)[0]
    if len(idx) > 1:
        M = base_logits[idx]
        r, c = linear_sum_assignment(-M)
        pred_hung[idx[r]] = c
acc_hung = balanced_accuracy_score(y_true, pred_hung)
print(f"2. Bipartite Hungarian with Recovered Clusters: {acc_hung*100:.2f}% (Gain: +{acc_hung*100 - acc_argmax*100:.2f}%)")

# 3. Hungarian with Oracle True Prompts
pred_oracle = pred_argmax.copy()
for p in np.unique(pids):
    idx = np.where(pids == p)[0]
    M = base_logits[idx]
    r, c = linear_sum_assignment(-M)
    pred_oracle[idx[r]] = c
acc_oracle = balanced_accuracy_score(y_true, pred_oracle)
print(f"3. Oracle True Prompt Hungarian: {acc_oracle*100:.2f}%")

