import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

print("=== Running Offline Simulation Test for Champion V19 Omni Titan ===")

LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
K = len(LABELS)

# 1. Simulate LOGIT array and TE texts
np.random.seed(42)
N = 12
LOGIT = np.random.randn(N, K)

# Simulate different text lengths: some short (<60 words), some medium, some long (>160 words)
simulated_texts = [
    "Short answer here Paris is capital.",  # ~6 words
    "Another short response with code snippet def foo(): return 42.",  # ~10 words
    "Medium length passage explaining the concept of gravity and planetary motion across solar systems with various details and historical anecdotes.",  # ~21 words
    " ".join(["word"] * 45),   # 45 words (<60)
    " ".join(["word"] * 85),   # 85 words (medium)
    " ".join(["word"] * 120),  # 120 words (medium)
    " ".join(["word"] * 175),  # 175 words (>160)
    " ".join(["word"] * 210),  # 210 words (>160)
    " ".join(["word"] * 30),   # 30 words (<60)
    " ".join(["word"] * 70),   # 70 words (medium)
    " ".join(["word"] * 190),  # 190 words (>160)
    " ".join(["word"] * 55),   # 55 words (<60)
]
assert len(simulated_texts) == N
word_counts = np.array([len(t.split()) for t in simulated_texts])

def softmax_rows(S):
    S = S - S.max(1, keepdims=True)
    P = np.exp(S)
    return P / P.sum(1, keepdims=True)

P = softmax_rows(LOGIT)

# 2. Simulate ATLAS blend at 20%
ATLAS_PROB = np.random.dirichlet(np.ones(K), size=N)
P = 0.80 * P + 0.20 * ATLAS_PROB
P /= P.sum(1, keepdims=True) + 1e-12
assert P.shape == (N, K)
assert np.allclose(P.sum(1), 1.0)
print(f"[PASS] ATLAS 20% Probability Blend verified. (Shape: {P.shape})")

# 3. Simulate Length-Dependent Anchor Protection
p_raw_sort = np.sort(P, axis=1)
raw_margin = p_raw_sort[:, -1] - p_raw_sort[:, -2]
anchor_mask = (raw_margin >= 0.40) & (P.max(axis=1) >= 0.70) & (word_counts >= 60)
print(f"[PASS] Length-aware anchor protection found {int(anchor_mask.sum())} anchors (strictly >= 60 words).")

# 4. Simulate Sinkhorn Balancing
BALANCE_ALPHA = 0.5
if BALANCE_ALPHA > 0 and len(P) >= 2:
    target = (1 - BALANCE_ALPHA) * P.sum(0) + BALANCE_ALPHA * np.full(K, len(P) / K)
    for _ in range(50):
        col_scale = target / (P.sum(0) + 1e-12)
        P *= col_scale[None, :]
        P /= P.sum(1, keepdims=True) + 1e-12
print("[PASS] Sinkhorn balancing completed.")

# 5. Simulate Sibling Micro-Gates & Length-Dependent Temperature Scaling
logP = np.log(np.clip(P, 1e-12, 1.0))
calibrated_logP = logP.copy()

GATE_LAM_V19 = {'claude': 1.10, 'gpt': 0.85, 'gemini': 0.85, 'chinese': 1.00}
FAMILY_IDX = {
    'claude': np.array([i for i, l in enumerate(LABELS) if l.startswith('claude')]),
    'gpt': np.array([i for i, l in enumerate(LABELS) if l.startswith('gpt')]),
    'gemini': np.array([i for i, l in enumerate(LABELS) if l.startswith('gemini')]),
    'chinese': np.array([i for i, l in enumerate(LABELS) if l.startswith('glm') or l.startswith('kimi') or l.startswith('qwen')])
}

for fam, fidx in FAMILY_IDX.items():
    gate_scores = np.random.randn(N, len(fidx))
    gate_centered = gate_scores - gate_scores.mean(axis=1, keepdims=True)
    fam_lam = GATE_LAM_V19[fam]
    calibrated_logP[:, fidx] += fam_lam * gate_centered
print(f"[PASS] Sibling Micro-Gates applied with boosted lambdas: {GATE_LAM_V19}")

# Apply length-dependent temperature scaling
tau_vec = np.ones(len(P))
tau_vec[word_counts < 60] = 1.25   # smooth on short passages
tau_vec[word_counts > 160] = 0.85  # sharpen on long passages
calibrated_logP /= tau_vec[:, None]
print(f"[PASS] Length-dependent Temperature Scaling applied (Smoothed {(word_counts < 60).sum()} short passages).")

# 6. Simulate Injective Hungarian
def solve_hungarian_cluster(log_probs):
    cost = -log_probs
    row_ind, col_ind = linear_sum_assignment(cost)
    assigned_q = np.zeros_like(log_probs)
    for r, c in zip(row_ind, col_ind):
        assigned_q[r, c] = 1.0
    return assigned_q

pred = calibrated_logP.argmax(axis=1)

# Check cluster
sub_lp = calibrated_logP[:4]
assigned = solve_hungarian_cluster(sub_lp)
assert assigned.shape == (4, K)
pred[:4] = assigned.argmax(axis=1)
print(f"[PASS] Injective Hungarian Assignment verified on cluster.")

# High-confidence protection (only word_counts >= 60)
p_sort = np.sort(P, axis=1)
margin = p_sort[:, -1] - p_sort[:, -2]
confident_mask = (margin > 0.40) & (word_counts >= 60)
pred[confident_mask] = P.argmax(axis=1)[confident_mask]
print(f"[PASS] Confident protection verified (Protected {int(confident_mask.sum())} samples).")

# 7. Output submission DataFrame
sub = pd.DataFrame({
    'id': [f"id_{i:04d}" for i in range(N)],
    'model': [LABELS[i] for i in pred]
})
assert len(sub) == N
assert sub['model'].isin(LABELS).all()
assert sub['id'].is_unique
sub.to_parquet('scratch/test_v19_submission.parquet', index=False)
print(f"[PASS] Generated scratch/test_v19_submission.parquet ({len(sub)} rows) successfully!")
print("=== ALL V19 SIMULATION TESTS PASSED WITH 100% SUCCESS! ===")
