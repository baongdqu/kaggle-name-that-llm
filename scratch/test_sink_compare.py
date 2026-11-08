import numpy as np

# Test Sinkhorn column scaling factors with V33 vs V34
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
fam_priors = fam_sizes / 18.0

true_y = np.repeat(np.arange(18), N // 18 + 1)[:N]
logits = np.random.randn(N, K) * 1.5
for i in range(N):
    logits[i, true_y[i]] += 2.5
P = np.exp(logits)
P /= P.sum(axis=1, keepdims=True)

# V33 Hierarchical
P_fam_base = np.zeros((N, len(fam_names)))
for f_idx, fname in enumerate(fam_names):
    P_fam_base[:, f_idx] = P[:, FAMILIES[fname]].sum(axis=1)

P_fam_v33 = np.power(P_fam_base, 1.22)
P_fam_v33 /= P_fam_v33.sum(axis=1, keepdims=True)
P_v33 = np.zeros_like(P)
for f_idx, fname in enumerate(fam_names):
    m_indices = FAMILIES[fname]
    intra = P[:, m_indices] / P[:, m_indices].sum(axis=1, keepdims=True)
    P_v33[:, m_indices] = intra * P_fam_v33[:, f_idx:f_idx+1]

# V34 Prior-Calibrated Hierarchical
odds = P_fam_base / fam_priors[None, :]
odds_sharp = np.power(odds, 1.20)
P_fam_v34 = odds_sharp * fam_priors[None, :]
P_fam_v34 /= P_fam_v34.sum(axis=1, keepdims=True)
P_v34 = np.zeros_like(P)
for f_idx, fname in enumerate(fam_names):
    m_indices = FAMILIES[fname]
    intra = P[:, m_indices] / P[:, m_indices].sum(axis=1, keepdims=True)
    P_v34[:, m_indices] = intra * P_fam_v34[:, f_idx:f_idx+1]

print("=== PRE-SINKHORN SUMS ===")
print("True Expected Column Sum:", N / 18.0)
print(f"V33 Qwen column sum: {P_v33[:, 17].sum():.2f}")
print(f"V34 Qwen column sum: {P_v34[:, 17].sum():.2f}")
print(f"V33 Claude-0 column sum: {P_v33[:, 0].sum():.2f}")
print(f"V34 Claude-0 column sum: {P_v34[:, 0].sum():.2f}")

# Now run Sinkhorn on both and compare accuracy
def run_sinkhorn(mat):
    M = mat.copy()
    for _ in range(100):
        M *= (N / K) / M.sum(axis=0, keepdims=True)
        M /= M.sum(axis=1, keepdims=True)
    return M

P_v33_sink = run_sinkhorn(P_v33)
P_v34_sink = run_sinkhorn(P_v34)
P_flat_sink = run_sinkhorn(P)

print("\n=== POST-SINKHORN ACCURACY ===")
print(f"Flat Sinkhorn (V30/V32 baseline): {(P_flat_sink.argmax(axis=1) == true_y).mean():.4f}")
print(f"V33 Flawed Hierarchical:          {(P_v33_sink.argmax(axis=1) == true_y).mean():.4f}")
print(f"V34 Prior-Calibrated Hierarchical: {(P_v34_sink.argmax(axis=1) == true_y).mean():.4f}")
