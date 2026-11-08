import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

train = pd.read_csv('data/train.csv')

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

v = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05)
Xc = v.fit_transform(train.text.values)
S = (Xc @ Xc.T).toarray()

# Check with thr = 0.15
Gte_015 = capped_agglo(S, 18, thr=0.15)
sz_015 = np.bincount(Gte_015)[Gte_015]
print("--- With thr = 0.15 ---")
print("Total clusters:", len(np.unique(Gte_015)))
print("Clusters with size >= 12:", len(np.unique(Gte_015[(sz_015 >= 12) & (sz_015 <= 18)])))
print("Rows in clusters with size >= 12:", ((sz_015 >= 12) & (sz_015 <= 18)).sum(), "out of", len(train))
print("Max cluster size:", sz_015.max())
print("Cluster size distribution:", pd.Series(sz_015).value_counts().sort_index())

# Check with thr = 0.02 (the non-perpassage setting!)
Gte_002 = capped_agglo(S, 18, thr=0.02)
sz_002 = np.bincount(Gte_002)[Gte_002]
print("\n--- With thr = 0.02 ---")
print("Total clusters:", len(np.unique(Gte_002)))
print("Clusters with size >= 12:", len(np.unique(Gte_002[(sz_002 >= 12) & (sz_002 <= 18)])))
print("Rows in clusters with size >= 12:", ((sz_002 >= 12) & (sz_002 <= 18)).sum(), "out of", len(train))

