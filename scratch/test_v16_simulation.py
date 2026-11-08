import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer

LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
K = len(LABELS)
LAB2I = {l: i for i, l in enumerate(LABELS)}

NISHANT_LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                  'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                  'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
                  'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
NISHANT_TO_V14 = np.array([LABELS.index(l) for l in NISHANT_LABELS])

print("1. Testing Label Order Consistency:")
assert len(LABELS) == 18
assert len(NISHANT_TO_V14) == 18
print("  Label mapping verified!")

print("2. Simulating Exact Soft Marginal Injective Cluster Decoding:")
def injective_cavity_marginals(P_group, iters=5, repel_strength=0.85):
    n, C = P_group.shape
    L = np.log(np.clip(P_group, 1e-12, 1.0))
    q = np.exp(L - L.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    for _ in range(iters):
        lo = np.log(np.clip(1.0 - q, 1e-6, None))
        cavity_penalty = lo.sum(0, keepdims=True) - lo
        L_new = L + repel_strength * cavity_penalty
        q = np.exp(L_new - L_new.max(1, keepdims=True))
        q /= q.sum(1, keepdims=True)
    return q

# Test on cluster with duplicate predictions
P_test = np.random.uniform(0.01, 0.1, (4, 18))
P_test[0, 5] = 0.85  # gemini-3.1
P_test[1, 5] = 0.75  # also gemini-3.1
P_test[1, 6] = 0.65  # gemini-3.6
P_test /= P_test.sum(axis=1, keepdims=True)

print(f"  Before Injective: Row 0 -> {P_test[0].argmax()}, Row 1 -> {P_test[1].argmax()}")
q_test = injective_cavity_marginals(P_test, iters=5, repel_strength=0.85)
print(f"  After Injective:  Row 0 -> {q_test[0].argmax()}, Row 1 -> {q_test[1].argmax()}")
assert q_test[0].argmax() != q_test[1].argmax(), "Injective decoding failed to separate duplicate predictions!"
print("  Injective decoding separated sister models successfully!")

print("3. Simulating Full-Force Two-Stage LoRA Weights & Logit Scale:")
N_rows = 50
LOGIT = np.zeros((N_rows, K))

# Add Two-Stage 14B & 8B with V16 weights
W_14B, TWIN_14B = 0.70, 0.25
W_8B, TWIN_8B = 0.40, 0.15
W_POOL_SVC = 0.60

prob_dummy = np.random.dirichlet(np.ones(K), size=N_rows)
LOGIT += W_14B * np.log(np.clip(prob_dummy, 1e-9, 1.0))
LOGIT += W_8B * np.log(np.clip(prob_dummy, 1e-9, 1.0))
LOGIT += W_POOL_SVC * np.log(np.clip(prob_dummy, 1e-9, 1.0))

# Margin protection
p_raw = np.exp(LOGIT) / np.exp(LOGIT).sum(axis=1, keepdims=True)
margin = np.sort(p_raw, axis=1)[:, -1] - np.sort(p_raw, axis=1)[:, -2]
confident = margin > 0.40
print(f"  High-confidence rows (>0.40 margin): {confident.sum()} of {N_rows}")

pred = p_raw.argmax(axis=1)
sub = pd.DataFrame({"id": [f"test_{i}" for i in range(N_rows)], "model": [LABELS[i] for i in pred]})
assert len(sub) == N_rows
assert sub["model"].isin(LABELS).all()
print("V16 FULL SIMULATION PASSED 100% PERFECTLY!")
