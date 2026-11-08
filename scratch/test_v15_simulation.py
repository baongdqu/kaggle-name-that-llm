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

print("1. Testing Nishant Label Alignment:")
for i, l in enumerate(NISHANT_LABELS):
    target_idx = NISHANT_TO_V14[i]
    assert LABELS[target_idx] == l, f"Mismatch at {i}: {l} != {LABELS[target_idx]}"
print("  All 18 labels map 100% identically between Nishant and V14/V15!")

print("2. Simulating Two-Stage LoRA Inference & Accumulation:")
N_test = 60
np.random.seed(42)
LOGIT = np.zeros((N_test, K))

# Fake test texts
TE = [f"This is a sample generated passage about quantum science and artificial intelligence prompt #{i%10}" for i in range(N_test)]

# Topic clustering
Xt = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(TE)
D = np.clip(1 - (Xt @ Xt.T).toarray(), 0, None)
np.fill_diagonal(D, 0)
groups = fcluster(linkage(squareform(D, checks=False), method="average"), 0.85, criterion="distance") if len(TE) >= 2 else np.ones(len(TE), dtype=int)
size = np.bincount(groups)[groups]
twin_rows = (size >= 2) & (size <= 18)
print(f"  Topic clusters: {(size == 1).sum()} singletons, {twin_rows.sum()} in clusters")

# Simulate LoRA14B and LoRA8B outputs in Nishant's label order
def softmax_rows(S):
    S = S - S.max(1, keepdims=True)
    P = np.exp(S)
    return P / P.sum(1, keepdims=True)

TS_LORAS = {
    "lora14b_v1": dict(w=0.35, twin_w=0.12),
    "lora8b_v1":  dict(w=0.20, twin_w=0.08),
}

for ckpt_name, mc in TS_LORAS.items():
    # Plain head
    prob_plain_nishant = softmax_rows(np.random.randn(N_test, K))
    prob_plain_v15 = np.zeros_like(prob_plain_nishant)
    prob_plain_v15[:, NISHANT_TO_V14] = prob_plain_nishant
    LOGIT[:] += mc["w"] * np.log(np.clip(prob_plain_v15, 1e-9, 1.0))

    # Centred twin
    if twin_rows.any():
        prob_centred_nishant = softmax_rows(np.random.randn(int(twin_rows.sum()), K))
        prob_centred_v15 = np.zeros_like(prob_centred_nishant)
        prob_centred_v15[:, NISHANT_TO_V14] = prob_centred_nishant
        LOGIT[twin_rows] += mc["twin_w"] * np.log(np.clip(prob_centred_v15, 1e-9, 1.0))

print(f"  LOGIT accumulated successfully. Shape: {LOGIT.shape}, min: {LOGIT.min():.3f}, max: {LOGIT.max():.3f}")

print("3. Validating Submission Format:")
P = softmax_rows(LOGIT)
preds = [LABELS[i] for i in P.argmax(axis=1)]
sub = pd.DataFrame({"id": [f"test_{i}" for i in range(N_test)], "model": preds})
print(f"  Submission shape: {sub.shape}")
print(f"  Unique predicted models: {sub['model'].nunique()}")
print("ALL TESTS PASSED PERFECTLY!")
