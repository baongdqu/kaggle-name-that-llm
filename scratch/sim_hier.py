import numpy as np
from scipy.optimize import linear_sum_assignment

# Test Two-Stage Optimal Transport vs Flat Sinkhorn
np.random.seed(42)
N = 1600
K = 18

FAMILIES = {
    'Anthropic': [0, 1, 2, 3, 4],
    'Google':    [5, 6, 7, 8],
    'Zhipu':     [9, 10],
    'OpenAI':    [11, 12, 13, 14, 15],
    'Moonshot':  [16],
    'Alibaba':   [17],
}
fam_names = list(FAMILIES.keys())
fam_sizes = np.array([len(FAMILIES[f]) for f in fam_names], dtype=np.float64)

# Generate synthetic true labels and realistic probabilities
true_y = np.repeat(np.arange(18), N // 18 + 1)[:N]
logits = np.random.randn(N, K) * 1.5
for i in range(N):
    logits[i, true_y[i]] += 2.5
P = np.exp(logits)
P /= P.sum(axis=1, keepdims=True)

# Accuracy of raw P
acc_raw = (P.argmax(axis=1) == true_y).mean()
print(f"Raw Accuracy: {acc_raw:.4f}")

# Method 1: Standard Flat Sinkhorn (V30 / V32)
P_flat = P.copy()
for _ in range(100):
    P_flat *= (N / K) / P_flat.sum(axis=0, keepdims=True)
    P_flat /= P_flat.sum(axis=1, keepdims=True)
acc_flat = (P_flat.argmax(axis=1) == true_y).mean()
print(f"Flat Sinkhorn Accuracy (V30/V32): {acc_flat:.4f}")

# Method 2: V33 Flawed Hierarchical (Power Sharpening before Flat Sinkhorn)
P_fam_base = np.zeros((N, len(fam_names)))
for f_idx, fname in enumerate(fam_names):
    P_fam_base[:, f_idx] = P[:, FAMILIES[fname]].sum(axis=1)
P_fam_sharp = np.power(P_fam_base, 1.22)
P_fam_sharp /= P_fam_sharp.sum(axis=1, keepdims=True)
P_v33 = np.zeros_like(P)
for f_idx, fname in enumerate(fam_names):
    m_indices = FAMILIES[fname]
    intra = P[:, m_indices] / P[:, m_indices].sum(axis=1, keepdims=True)
    P_v33[:, m_indices] = intra * P_fam_sharp[:, f_idx:f_idx+1]
for _ in range(100):
    P_v33 *= (N / K) / P_v33.sum(axis=0, keepdims=True)
    P_v33 /= P_v33.sum(axis=1, keepdims=True)
acc_v33 = (P_v33.argmax(axis=1) == true_y).mean()
print(f"V33 Flawed Hierarchical Accuracy: {acc_v33:.4f}")

# Method 3: Two-Stage Mass-Preserving Optimal Transport (V34)
# Stage 1: Family Sinkhorn to exactly match family target marginals
T_fam = fam_sizes * (N / K) # [444.4, 355.6, 177.8, 444.4, 88.9, 88.9]
P_fam_opt = P_fam_base.copy()
for _ in range(100):
    P_fam_opt *= T_fam / P_fam_opt.sum(axis=0, keepdims=True)
    P_fam_opt /= P_fam_opt.sum(axis=1, keepdims=True)

# Stage 2: Intra-Family Balanced Sinkhorn
P_v34 = np.zeros_like(P)
for f_idx, fname in enumerate(fam_names):
    m_indices = FAMILIES[fname]
    nf = len(m_indices)
    if nf == 1:
        P_v34[:, m_indices[0]] = P_fam_opt[:, f_idx]
    else:
        # Intra-family probabilities
        Q = P[:, m_indices].copy()
        target_col = np.full(nf, N / K)
        for _ in range(100):
            Q *= target_col / Q.sum(axis=0, keepdims=True)
            Q /= Q.sum(axis=1, keepdims=True)
        # Scale row by P_fam_opt
        P_v34[:, m_indices] = Q * P_fam_opt[:, f_idx:f_idx+1]

acc_v34 = (P_v34.argmax(axis=1) == true_y).mean()
print(f"V34 Two-Stage Mass-Preserving OT Accuracy: {acc_v34:.4f}")

# Check final column sums of V34:
print("V34 col sums min:", P_v34.sum(axis=0).min(), "max:", P_v34.sum(axis=0).max(), "target:", N/K)
