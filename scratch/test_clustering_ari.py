import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
import time

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

print("Loading data/train.csv...")
df = pd.read_csv('data/train.csv')
y_true = df['prompt_id'].values

print("\n--- 1. Baseline TF-IDF ---")
v1 = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05)
X1 = v1.fit_transform(df['text'].values)
S1 = (X1 @ X1.T).toarray()
for thr in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
    g1 = capped_agglo(S1, 18, thr)
    ari = adjusted_rand_score(y_true, g1)
    n_cl = len(np.unique(g1))
    sizes = np.bincount(g1)
    dense = (sizes >= 5).sum()
    print(f"Baseline thr={thr:.2f} -> ARI: {ari:.4f} | {n_cl} clusters | dense>=5: {dense} clusters")

print("\n--- 2. Multi-View TF-IDF (Word + Char-wb) ---")
v_word = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.10, ngram_range=(1, 2))
v_char = TfidfVectorizer(sublinear_tf=True, analyzer='char_wb', min_df=3, max_df=0.15, ngram_range=(3, 5))
X_word = v_word.fit_transform(df['text'].values)
X_char = v_char.fit_transform(df['text'].values)

S_word = (X_word @ X_word.T).toarray()
S_char = (X_char @ X_char.T).toarray()

for alpha in [0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]:
    S_blend = (1 - alpha) * S_word + alpha * S_char
    for thr in [0.02, 0.04, 0.06, 0.08, 0.10]:
        g2 = capped_agglo(S_blend, 18, thr)
        ari = adjusted_rand_score(y_true, g2)
        if ari > 0.80 or thr == 0.04:
            n_cl = len(np.unique(g2))
            sizes = np.bincount(g2)
            dense = (sizes >= 5).sum()
            print(f"Blend alpha={alpha:.1f} (char) thr={thr:.2f} -> ARI: {ari:.4f} | {n_cl} clusters | dense>=5: {dense}")

