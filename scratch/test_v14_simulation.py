import numpy as np
from sklearn.svm import LinearSVC
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
K = len(LABELS)
LAB2I = {l: i for i, l in enumerate(LABELS)}

FAMILY_IDX = {
    'claude': np.array([i for i, l in enumerate(LABELS) if l.startswith('claude')]),
    'gpt': np.array([i for i, l in enumerate(LABELS) if l.startswith('gpt')]),
    'gemini': np.array([i for i, l in enumerate(LABELS) if l.startswith('gemini')]),
    'chinese': np.array([i for i, l in enumerate(LABELS) if l.startswith('glm') or l.startswith('kimi') or l.startswith('qwen')])
}

print("1. Validating FAMILY_IDX:")
for k, v in FAMILY_IDX.items():
    print(f"  {k}: {len(v)} models -> {[LABELS[i] for i in v]}")

assert sum(len(v) for v in FAMILY_IDX.values()) == 18

print("2. Simulating Cell 16 Soft Cross-Family Calibration:")
N_test = 50
LOGIT = np.random.randn(N_test, K)
P = np.exp(LOGIT) / np.exp(LOGIT).sum(axis=1, keepdims=True)

# Fake GATES
GATES = {}
for fam, fidx in FAMILY_IDX.items():
    d = np.random.randn(N_test, len(fidx))
    GATES[fam] = d

# Fake global 18 gate
d_global = np.random.randn(N_test, K)
GATES['global_18'] = d_global - d_global.mean(axis=1, keepdims=True)

# V14 Cross-Family Soft Posterior Calibration
logP = np.log(np.clip(P, 1e-12, 1.0))
calibrated_logP = logP.copy()

# 1. Apply Universal 18-class global calibration
if 'global_18' in GATES:
    lam_global = 0.35
    calibrated_logP += lam_global * GATES['global_18']

# 2. Apply Intra-family micro-calibration for all 4 families
GATE_LAM_V14 = {'claude': 0.80, 'gpt': 0.60, 'gemini': 0.60, 'chinese': 0.80}
for fam, fidx in FAMILY_IDX.items():
    if fam not in GATES:
        continue
    gate_scores = GATES[fam]
    gate_centered = gate_scores - gate_scores.mean(axis=1, keepdims=True)
    fam_lam = GATE_LAM_V14.get(fam, 0.60)
    calibrated_logP[:, fidx] += fam_lam * gate_centered

# 3. High-confidence ensemble protection
p_sort = np.sort(P, axis=1)
margin = p_sort[:, -1] - p_sort[:, -2]
confident_mask = margin > 0.35

pred = calibrated_logP.argmax(axis=1)
pred[confident_mask] = P.argmax(axis=1)[confident_mask]

flips = int((pred != P.argmax(1)).sum())
print(f"  V14 modified {flips} of {len(P)} predictions.")
print("  All predictions valid:", all(0 <= p < 18 for p in pred))
print("Simulation passed with flying colors!")
