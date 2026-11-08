import numpy as np
import pandas as pd
import os
import time

LABELS = [
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra',
    'gpt-6-astra', 'kimi-k3', 'qwen3.8-max'
]
assert len(LABELS) == 18

print("--- Testing Champion V17 Simulation Offline ---")

# 1. Create dummy test set (with clusters of sizes 2, 3, 5, and singles)
texts = [
    "Write a python script to calculate fibonacci numbers.",
    "Sure! Here is the python script to calculate fibonacci numbers accurately.",
    "Write a quick recipe for chocolate cake with sugar.",
    "Here is an easy and delicious chocolate cake recipe for you.",
    "Another chocolate cake recipe made with rich dark chocolate.",
    "Explain general relativity in simple layman terms.",
    "General relativity is Einstein's theory of gravitation and curved spacetime.",
    "Think of spacetime like a rubber sheet stretched tight.",
    "When a heavy ball sits on the sheet, it curves the fabric.",
    "Mass tells spacetime how to curve, and curvature tells mass how to move.",
    "A single question about quantum mechanics."
]
test_df = pd.DataFrame({
    'id': [f"test_{i:04d}" for i in range(len(texts))],
    'text': texts
})
TE = test_df['text'].tolist()
N = len(TE)
print(f"Generated {N} test rows.")

# 2. Simulate baseline logit from Two-Stage Qwen (14B + 8B) + 35k SVC
np.random.seed(42)
LOGIT = np.random.randn(N, 18) * 0.8

# 3. Simulate OLMo2 Native-24 Backbone
olmo_classes = np.array(LABELS)
simulated_olmo_logp = np.random.randn(N, 18)
reorder = np.asarray([list(olmo_classes).index(lbl) for lbl in LABELS], dtype=int)
olmo_ordered_logp = simulated_olmo_logp[:, reorder]
W_OLMO = 0.20
olmo_centered = olmo_ordered_logp - np.mean(olmo_ordered_logp, axis=1, keepdims=True)
LOGIT += W_OLMO * olmo_centered
print("OLMo2 backbone logit integration simulated.")

# 4. Mega-Gate simulation
def softmax_rows(X):
    e = np.exp(X - np.max(X, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

P = softmax_rows(LOGIT)
calibrated_logP = np.log(np.clip(P, 1e-12, 1.0))

# 5. Hybrid Injective Decoding
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

def exclusive_marginals_dp(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    p = p / np.clip(p.sum(1, keepdims=True), 1e-12, None)
    n, k = p.shape
    if n <= 1:
        return p.copy()
    if n > k or n > 6:
        return None
    full = (1 << n) - 1
    f = np.zeros((k + 1, 1 << n))
    b = np.zeros_like(f)
    f[0, 0] = 1.0
    b[k, 0] = 1.0
    for c in range(k):
        f[c + 1] = f[c]
        for mask in range(1, full + 1):
            for i in range(n):
                if mask & (1 << i):
                    f[c + 1, mask] += p[i, c] * f[c, mask ^ (1 << i)]
    for c in range(k - 1, -1, -1):
        b[c] = b[c + 1]
        for mask in range(1, full + 1):
            for i in range(n):
                if mask & (1 << i):
                    b[c, mask] += p[i, c] * b[c + 1, mask ^ (1 << i)]
    z = f[k, full]
    if z <= np.finfo(float).tiny:
        return p.copy()
    q = np.zeros_like(p)
    for i in range(n):
        remaining = full ^ (1 << i)
        for c in range(k):
            subset = remaining
            while True:
                q[i, c] += p[i, c] * f[c, subset] * b[c + 1, remaining ^ subset] / z
                if subset == 0:
                    break
                subset = (subset - 1) & remaining
    return q / np.clip(q.sum(1, keepdims=True), 1e-12, None)

def cavity_marginals(p, iters=5, repel_strength=0.85):
    L = np.log(np.clip(p, 1e-12, 1.0))
    q = np.exp(L - L.max(1, keepdims=True))
    q /= np.clip(q.sum(1, keepdims=True), 1e-12, None)
    for _ in range(iters):
        lo = np.log(np.clip(1.0 - q, 1e-6, None))
        cavity_penalty = lo.sum(0, keepdims=True) - lo
        L_new = L + repel_strength * cavity_penalty
        q = np.exp(L_new - L_new.max(1, keepdims=True))
        q /= np.clip(q.sum(1, keepdims=True), 1e-12, None)
    return q

def hybrid_injective_decoding(P_group):
    n = len(P_group)
    if n <= 1:
        return P_group
    repel_weight = min(0.95, 0.40 * np.log2(n) + 0.30)
    if n <= 6:
        exact_q = exclusive_marginals_dp(P_group)
        if exact_q is not None and np.isfinite(exact_q).all():
            return (1.0 - repel_weight) * P_group + repel_weight * exact_q
    cavity_q = cavity_marginals(P_group, iters=5, repel_strength=repel_weight)
    return (1.0 - repel_weight) * P_group + repel_weight * cavity_q

P_calibrated = softmax_rows(calibrated_logP)
Xt_dec = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(TE)
D_dec = np.clip(1 - (Xt_dec @ Xt_dec.T).toarray(), 0, None)
np.fill_diagonal(D_dec, 0)
groups_dec = fcluster(linkage(squareform(D_dec, checks=False), method="average"), 0.85, criterion="distance")
size_dec = np.bincount(groups_dec)[groups_dec]
cluster_rows = (size_dec >= 2) & (size_dec <= 18)

if cluster_rows.any():
    for g in np.unique(groups_dec[cluster_rows]):
        g_idx = np.where(groups_dec == g)[0]
        if len(g_idx) >= 2 and len(g_idx) <= 18:
            P_calibrated[g_idx] = hybrid_injective_decoding(P_calibrated[g_idx])
    print(f'Hybrid Injective decoding applied to {int(cluster_rows.sum())} clustered rows in {len(np.unique(groups_dec[cluster_rows]))} groups')

calibrated_logP = np.log(np.clip(P_calibrated, 1e-12, 1.0))

p_sort = np.sort(P, axis=1)
margin = p_sort[:, -1] - p_sort[:, -2]
confident_mask = margin > 0.40

pred = calibrated_logP.argmax(axis=1)
pred[confident_mask] = P.argmax(axis=1)[confident_mask]

# 6. Check final output
pred_labels = [LABELS[idx] for idx in pred]
sub_df = pd.DataFrame({
    'id': test_df['id'],
    'model': pred_labels
})
sub_df.to_parquet('scratch/test_v17_sub.parquet', index=False)

assert len(sub_df) == len(test_df)
assert set(sub_df['model']).issubset(set(LABELS))
assert not sub_df['id'].isna().any()
assert not sub_df['model'].isna().any()

print("\nSUCCESS: V17 Simulation finished with 100% validity!")
print(sub_df.head(10))
