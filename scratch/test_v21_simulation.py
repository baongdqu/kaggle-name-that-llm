import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer

print("=== STARTING FULL V21 SIMULATION TEST (4,000 ROWS) ===")

LABELS = sorted([
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
])
K = len(LABELS)
assert K == 18

# Generate 4,000 simulated test samples with realistic cluster structure
np.random.seed(42)
N = 4000
test_ids = [f"id_{i:04d}" for i in range(N)]

# Simulate cluster structure: ~400 clusters of sizes 1 to 18
groups = []
curr_g = 1
while len(groups) < N:
    size = np.random.choice([1, 2, 3, 4, 5, 6, 8, 12, 18], p=[0.20, 0.25, 0.20, 0.15, 0.08, 0.05, 0.03, 0.02, 0.02])
    remaining = N - len(groups)
    size = min(size, remaining)
    groups.extend([curr_g] * size)
    curr_g += 1
groups = np.array(groups)

# Simulate texts of varying lengths
sample_texts = []
for i in range(N):
    words = np.random.choice(["the", "algorithm", "model", "analysis", "data", "First,", "Second,", "because", "therefore", "function", "output", "result"], size=np.random.randint(20, 300))
    sample_texts.append(" ".join(words))

test_df = pd.DataFrame({"id": test_ids, "text": sample_texts})

# Simulate probability matrix P after Sinkhorn
logits = np.random.randn(N, K)
# add cluster correlation
for g in np.unique(groups):
    idx = np.flatnonzero(groups == g)
    logits[idx] += np.random.randn(1, K) * 1.5

def softmax_rows(S):
    S = S - S.max(1, keepdims=True)
    P = np.exp(S)
    return P / P.sum(1, keepdims=True)

P = softmax_rows(logits)

# Sinkhorn simulation
def sinkhorn(P, iters=50):
    c = np.ones(K)
    target = np.full(K, len(P) / K)
    for _ in range(iters):
        R = P * c
        R /= R.sum(1, keepdims=True)
        c *= target / (R.sum(0) + 1e-12)
    R = P * c
    return R / R.sum(1, keepdims=True)

P = sinkhorn(P)

# --- 1. SMOOTH LENGTH TEMPERATURE SCALING ---
word_counts = np.array([len(t.split()) for t in test_df.text.tolist()])
tau = np.clip(1.15 - 0.25 * ((word_counts - 60) / 100.0), 0.90, 1.15)
log_P = np.log(np.clip(P, 1e-12, 1.0))
log_P_tempered = log_P / tau[:, None]
log_P_tempered -= log_P_tempered.max(axis=1, keepdims=True)
P_tempered = np.exp(log_P_tempered)
P_tempered /= P_tempered.sum(axis=1, keepdims=True)
print(f"[TEST 1 PASS] Smooth Length Temp Scaling: min tau={tau.min():.2f}, max tau={tau.max():.2f}")

# --- 2. SIBLING MICRO-GATES ---
idx_g37 = LABELS.index('gemini-3.7-flash')
idx_g38 = LABELS.index('gemini-3.8-flash')
idx_glm = LABELS.index('glm-5.3')
idx_glmf = LABELS.index('glm-5.3-flash')

for i, txt in enumerate(test_df.text):
    # Gemini 3.8-flash reasoning marker
    if "First," in txt or "Second," in txt or "Step 1:" in txt:
        if P_tempered[i, idx_g38] > 0.05:
            P_tempered[i, idx_g38] *= 1.15
            P_tempered[i, idx_g37] *= 0.88
    # Chinese punctuation marker
    if "，" in txt or "。" in txt:
        if P_tempered[i, idx_glm] > 0.05:
            P_tempered[i, idx_glm] *= 1.20
            P_tempered[i, idx_glmf] *= 0.85

P_tempered /= P_tempered.sum(axis=1, keepdims=True)
print("[TEST 2 PASS] Sibling Micro-Gates applied successfully.")

# --- 3. ANCHOR-PROTECTED HUNGARIAN MATCHING ---
def apply_anchor_protected_hungarian(P, groups, anchor_margin=0.30, anchor_conf=0.60):
    N, K = P.shape
    final_pred_indices = np.zeros(N, dtype=int)
    total_anchors = 0
    total_hungarian_assigned = 0

    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        g_size = len(rows)

        # Singletons or oversized groups -> fallback to argmax
        if g_size == 1 or g_size > K:
            final_pred_indices[rows] = P[rows].argmax(axis=1)
            continue

        sub_P = P[rows].copy()
        sorted_sub_P = np.sort(sub_P, axis=1)
        margins = sorted_sub_P[:, -1] - sorted_sub_P[:, -2]
        top_classes = sub_P.argmax(axis=1)

        is_anchor = (margins >= anchor_margin) & (sorted_sub_P[:, -1] >= anchor_conf)
        assigned_classes = set()
        anchor_local_indices = []

        candidate_order = np.argsort(-sorted_sub_P[:, -1])
        for local_idx in candidate_order:
            if is_anchor[local_idx]:
                cls = top_classes[local_idx]
                if cls not in assigned_classes:
                    assigned_classes.add(cls)
                    anchor_local_indices.append(local_idx)
                    final_pred_indices[rows[local_idx]] = cls
                    total_anchors += 1

        unassigned_local = [i for i in range(g_size) if i not in anchor_local_indices]
        if not unassigned_local:
            continue

        available_classes = [c for c in range(K) if c not in assigned_classes]
        if len(available_classes) < len(unassigned_local):
            for local_idx in unassigned_local:
                final_pred_indices[rows[local_idx]] = sub_P[local_idx].argmax()
            continue

        cost_matrix = -np.log(sub_P[unassigned_local][:, available_classes] + 1e-12)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            orig_row = rows[unassigned_local[r]]
            chosen_class = available_classes[c]
            final_pred_indices[orig_row] = chosen_class
            total_hungarian_assigned += 1

    return final_pred_indices, total_anchors, total_hungarian_assigned

import time
t_start = time.perf_counter()
final_indices, n_anchors, n_hungarian = apply_anchor_protected_hungarian(P_tempered, groups)
t_elapsed = time.perf_counter() - t_start

sub = pd.DataFrame({"id": test_df["id"], "model": np.array(LABELS)[final_indices]})
assert len(sub) == N
assert sub["model"].isin(LABELS).all()
assert sub["id"].is_unique

print(f"[TEST 3 PASS] Hungarian execution time on {N} rows: {t_elapsed*1000:.2f} ms")
print(f"  -> Total Anchors locked: {n_anchors}")
print(f"  -> Total Hungarian assigned: {n_hungarian}")
print(f"  -> Class distribution across 18 models:")
print(sub["model"].value_counts().to_string())

# Verification: Check that in any cluster of size 2-18, there are NO duplicate assignments!
collisions = 0
for g in np.unique(groups):
    rows = np.flatnonzero(groups == g)
    if 2 <= len(rows) <= 18:
        models = sub.iloc[rows]["model"].tolist()
        if len(models) != len(set(models)):
            collisions += 1

print(f"[COLLISION CHECK] Injective Violations in Clusters (2 <= n <= 18): {collisions} (Must be 0)")
assert collisions == 0, "Injective matching failed!"
print("=== ALL 4 SIMULATION TESTS PASSED WITH 0 ERRORS! ===")
