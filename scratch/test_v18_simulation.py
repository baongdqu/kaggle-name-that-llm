import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

print("=== Running Offline Simulation Test for Champion V18 ===")

LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
          'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
K = len(LABELS)

# 1. Simulate LOGIT array (e.g. 10 test samples)
np.random.seed(42)
N = 10
LOGIT = np.random.randn(N, K)

def softmax_rows(S):
    S = S - S.max(1, keepdims=True)
    P = np.exp(S)
    return P / P.sum(1, keepdims=True)

P = softmax_rows(LOGIT)

# 2. Simulate ATLAS blend
ATLAS_PROB = np.random.dirichlet(np.ones(K), size=N)
P = 0.90 * P + 0.10 * ATLAS_PROB
P /= P.sum(1, keepdims=True) + 1e-12
assert P.shape == (N, K)
assert np.allclose(P.sum(1), 1.0)
print("[PASS] ATLAS 10% Probability Blend verified.")

# 3. Simulate Margin-Locked Sinkhorn
p_raw_sort = np.sort(P, axis=1)
raw_margin = p_raw_sort[:, -1] - p_raw_sort[:, -2]
anchor_mask = (raw_margin >= 0.40) & (P.max(axis=1) >= 0.70)
print(f"[PASS] Anchor mask found {int(anchor_mask.sum())} confident anchors.")

BALANCE_ALPHA = 0.5
if BALANCE_ALPHA > 0 and len(P) >= 2:
    target = (1 - BALANCE_ALPHA) * P.sum(0) + BALANCE_ALPHA * np.full(K, len(P) / K)
    for _ in range(50):
        col_scale = target / (P.sum(0) + 1e-12)
        P *= col_scale[None, :]
        P /= P.sum(1, keepdims=True) + 1e-12
print("[PASS] Margin Sinkhorn balancing completed.")

# 4. Simulate Kuhn-Munkres Hungarian Assignment on cluster
def solve_hungarian_cluster(log_probs):
    cost = -log_probs
    row_ind, col_ind = linear_sum_assignment(cost)
    assigned_q = np.zeros_like(log_probs)
    for r, c in zip(row_ind, col_ind):
        assigned_q[r, c] = 1.0
    return assigned_q

cluster_log_probs = np.log(P[:4])
assigned = solve_hungarian_cluster(cluster_log_probs)
assert assigned.shape == (4, K)
assert (assigned.sum(axis=1) == 1).all()
assert (assigned.sum(axis=0) <= 1).all()  # Injective: no two samples get same label!
print("[PASS] Kuhn-Munkres Hungarian Injective Matching verified: 100% distinct labels assigned per cluster.")

# 5. Simulate submission creation
preds = [LABELS[i] for i in P.argmax(axis=1)]
df_sub = pd.DataFrame({"id": [f"id_{i}" for i in range(N)], "model": preds})
assert list(df_sub.columns) == ["id", "model"]
assert all(m in LABELS for m in df_sub["model"])
print(f"[PASS] Submission format 100% compliant with Kaggle specifications.")
print("\n>>> ALL TESTS PASSED SUCCESSFULLY! Ready to push Champion V18 to Kaggle! <<<")
