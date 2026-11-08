import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
import time

train = pd.read_csv('data/train.csv')
y_true = train.prompt_id.values

def fast_capped_agglo(S, cap=18, thr=0.02):
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

print("Testing baseline TF-IDF on a 40% slice (similar to Public LB ~1080 rows)...")
np.random.seed(42)
slice_idx = np.where(np.random.rand(len(train)) < 0.40)[0]
slice_texts = train.text.values[slice_idx]
slice_y = y_true[slice_idx]
print(f"Slice size: {len(slice_idx)} rows, {len(np.unique(slice_y))} true prompts")

t0 = time.time()
v1 = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05)
X1 = v1.fit_transform(slice_texts)
S1 = (X1 @ X1.T).toarray()
print(f"TF-IDF computed in {time.time()-t0:.2f}s, vocabulary size: {X1.shape[1]}")

for thr in [0.01, 0.02, 0.04, 0.08, 0.15]:
    t1 = time.time()
    g = fast_capped_agglo(S1, 18, thr)
    ari = adjusted_rand_score(slice_y, g)
    sz = np.bincount(g)
    dense5 = (sz >= 5).sum()
    print(f"thr={thr:.2f} | ARI: {ari:.4f} | Dense>=5 clusters: {dense5} | Time: {time.time()-t1:.2f}s")

