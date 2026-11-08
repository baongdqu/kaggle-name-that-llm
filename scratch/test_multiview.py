import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
import time

train = pd.read_csv('data/train.csv')
y_true = train.prompt_id.values

def fast_capped_agglo(S, cap=18, thr=0.06):
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

np.random.seed(42)
slice_idx = np.where(np.random.rand(len(train)) < 0.40)[0]
slice_texts = train.text.values[slice_idx]
slice_y = y_true[slice_idx]

print("--- Testing TF-IDF Feature Sets on 40% slice ---")

# Setup 1: Baseline word TF-IDF
v1 = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05)
X1 = v1.fit_transform(slice_texts)
S1 = (X1 @ X1.T).toarray()

# Setup 2: Expanded max_df (0.10) with n-grams (1, 2)
v2 = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.10, ngram_range=(1, 2))
X2 = v2.fit_transform(slice_texts)
S2 = (X2 @ X2.T).toarray()

# Setup 3: Word + Char-wb (3, 4) with max_features to keep it fast
v3_char = TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', min_df=3, max_features=25000, ngram_range=(3, 4))
X3_char = v3_char.fit_transform(slice_texts)
S3_char = (X3_char @ X3_char.T).toarray()

S_blend = 0.7 * S2 + 0.3 * S3_char

for name, S in [('Baseline (max_df 0.05)', S1), ('Expanded (1,2) max_df 0.10', S2), ('Hybrid Word+Char', S_blend)]:
    for thr in [0.05, 0.06, 0.07, 0.08, 0.09]:
        g = fast_capped_agglo(S, 18, thr)
        ari = adjusted_rand_score(slice_y, g)
        sz = np.bincount(g)
        dense5 = (sz >= 5).sum()
        print(f"{name:26s} | thr={thr:.2f} | ARI: {ari:.4f} | Dense>=5: {dense5}", flush=True)

