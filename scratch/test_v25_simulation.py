import time
import numpy as np
import pandas as pd

LABELS = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]
K = len(LABELS)

print("=== STEP 1: SIMULATE 4,000 TEST SAMPLES DECISION FLOW FOR V25 ===")
N_TEST = 4000
np.random.seed(42)

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

# 1. Main logits simulation with V25 weights:
# W_SVC = 0.85, Qwen3_14B = 0.60, Qwen2.5_14B = 0.25, DeBERTa = 0.30
svc_logits = np.random.randn(N_TEST, K) * 1.8
qwen14_logits = np.random.randn(N_TEST, K) * 2.0
qwen25_logits = np.random.randn(N_TEST, K) * 1.7
deb_logits = np.random.randn(N_TEST, K) * 1.5

main_logits = 0.85 * svc_logits + 0.60 * qwen14_logits + 0.25 * qwen25_logits + 0.30 * deb_logits
_main_raw = softmax(main_logits)

# 2. Comp stack logits simulation (12 models)
comp_logits = np.random.randn(N_TEST, K) * 1.6
_comp = softmax(comp_logits)

# V25 Blending: 0.75 * _main_raw + 0.25 * _comp (was 0.85 / 0.15)
P = 0.75 * _main_raw + 0.25 * _comp
print(f"P after Comp Blend: finite={np.isfinite(P).all()}, sum={np.allclose(P.sum(1), 1)}")

# 3. ATLAS Likelihood Rescoring on margin < 0.40 (was 0.30) with weight 0.20 (was 0.10)
_margins = np.sort(P, axis=1)
_margins = _margins[:, -1] - _margins[:, -2]
_atlas_rows = np.flatnonzero(_margins < 0.40)
print(f"ATLAS active rows: {len(_atlas_rows)} / {N_TEST} ({len(_atlas_rows)/N_TEST*100:.1f}%)")

atlas_prob = softmax(np.random.randn(len(_atlas_rows), K) * 2.2)
P[_atlas_rows] = 0.80 * P[_atlas_rows] + 0.20 * atlas_prob
print(f"P after ATLAS: finite={np.isfinite(P).all()}, sum={np.allclose(P.sum(1), 1)}")

# 4. Night24 Spectral 4,404 Heads: 0.90 * P + 0.10 * hp
hp = softmax(np.random.randn(N_TEST, K) * 1.5)
P = 0.90 * P + 0.10 * hp
print(f"P after Spectral: finite={np.isfinite(P).all()}, sum={np.allclose(P.sum(1), 1)}")

# 5. Sibling Micro-Gates
idx_g37 = LABELS.index('gemini-3.7-flash')
idx_g38 = LABELS.index('gemini-3.8-flash')
idx_glm = LABELS.index('glm-5.3')
idx_glmf = LABELS.index('glm-5.3-flash')

mock_texts = [
    "First, Step 1: analyze and In conclusion, answer is here.",
    "这是一段包含「引号」与中文句号。的文本",
    "Normal sentence without special markers."
] * (N_TEST // 3 + 1)
mock_texts = mock_texts[:N_TEST]

for _i, _txt in enumerate(mock_texts):
    if any(m in _txt for m in ['First,', 'Second,', 'Step 1:', '1. ', 'In conclusion,']):
        if P[_i, idx_g38] > 0.04:
            P[_i, idx_g38] *= 1.12
            P[_i, idx_g37] *= 0.90
    if any(m in _txt for m in ['，', '。', '「', '」', '、', '《', '》']):
        if P[_i, idx_glm] > 0.04:
            P[_i, idx_glm] *= 1.15
            P[_i, idx_glmf] *= 0.88

P = P / P.sum(1, keepdims=True)
print(f"P after Micro-Gates: finite={np.isfinite(P).all()}, sum={np.allclose(P.sum(1), 1)}")

# 6. Terminal Sinkhorn Purity (300 iters)
def sinkhorn(P, iters=300):
    c = np.ones(K)
    target = np.full(K, len(P) / K)
    for _ in range(iters):
        R = P * c
        R /= R.sum(1, keepdims=True)
        c *= target / R.sum(0)
    R = P * c
    return R / R.sum(1, keepdims=True)

t0 = time.time()
P_balanced = sinkhorn(P, iters=300)
elapsed_sinkhorn = time.time() - t0

preds = P_balanced.argmax(axis=1)
counts = pd.Series(preds).value_counts().sort_index()

print(f"\nSinkhorn converged in {elapsed_sinkhorn*1000:.2f} ms")
print(f"Column sum deviation: {np.abs(P_balanced.sum(0) - N_TEST/K).max():.8f}")
print("Predicted class distribution across 18 classes:")
for label_idx, count in counts.items():
    print(f"  {LABELS[label_idx]:25s}: {count} ({count/N_TEST*100:.2f}%)")

assert np.abs(P_balanced.sum(0) - N_TEST/K).max() < 1e-4, "Sinkhorn failed to converge to uniform totals!"
assert counts.min() >= 180 and counts.max() <= 270, "Unbalanced predictions detected!"
print("\nV25 SIMULATION TEST: ALL CHECKS PASSED 100%!")
