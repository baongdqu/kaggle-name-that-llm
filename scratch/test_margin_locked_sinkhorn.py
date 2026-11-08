import numpy as np

# Simulate test probabilities for 4000 samples across 18 classes
np.random.seed(42)
N = 4000
K = 18

logits = np.random.randn(N, K)
# Make some samples very confident
logits[:2000, 0] += 3.0
logits[2000:3000, 1] += 2.5

P = np.exp(logits - logits.max(axis=1, keepdims=True))
P /= P.sum(axis=1, keepdims=True)

# Test Margin-Locked Sinkhorn
top2_idx = np.argsort(P, axis=1)[:, -2:]
p_top1 = P[np.arange(N), top2_idx[:, 1]]
p_top2 = P[np.arange(N), top2_idx[:, 0]]
margins = p_top1 - p_top2

MARGIN_THRESH = 0.35
anchor_mask = (margins > MARGIN_THRESH) | (p_top1 > 0.60)
num_anchors = int(anchor_mask.sum())
num_malleable = N - num_anchors

pred = P.argmax(axis=1)
locked_preds = pred[anchor_mask]

# Compute target quota for malleable samples
target_total = np.full(K, N / K)
locked_counts = np.bincount(locked_preds, minlength=K)
residual_target = np.maximum(1.0, target_total - locked_counts)
residual_target = residual_target / residual_target.sum() * num_malleable

print(f"Total samples: {N}")
print(f"Locked high-confidence anchors: {num_anchors} ({num_anchors/N*100:.1f}%)")
print(f"Malleable borderline samples: {num_malleable} ({num_malleable/N*100:.1f}%)")

# Run soft Sinkhorn only on malleable samples
P_malleable = P[~anchor_mask].copy()
for _ in range(50):
    P_malleable *= (residual_target / (P_malleable.sum(axis=0) + 1e-12))[None, :]
    P_malleable /= P_malleable.sum(axis=1, keepdims=True) + 1e-12

malleable_preds = P_malleable.argmax(axis=1)

final_preds = pred.copy()
final_preds[~anchor_mask] = malleable_preds

print(f"Final predictions shape: {final_preds.shape}")
print(f"Anchor predictions unchanged: {np.array_equal(final_preds[anchor_mask], locked_preds)}")
print("Class distribution before:", np.bincount(pred, minlength=K))
print("Class distribution after:", np.bincount(final_preds, minlength=K))
