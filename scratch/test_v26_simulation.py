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

def get_family(name):
    for f in ('gpt', 'gemini', 'claude', 'glm', 'qwen', 'kimi'):
        if name.startswith(f):
            return f
    return name

FAMS = [get_family(l) for l in LABELS]

print("=== STEP 1: SIMULATE 4,000 TEST SAMPLES DECISION FLOW FOR V26 ===")
N_TEST = 4000
np.random.seed(42)

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

# 1. Main logits with golden V24 weights: W_SVC=1.0, 14B=0.4, 25_14B=0.2, DeBERTa=0.25
svc_logits = np.random.randn(N_TEST, K) * 1.8
qwen14_logits = np.random.randn(N_TEST, K) * 2.0
qwen25_logits = np.random.randn(N_TEST, K) * 1.7
deb_logits = np.random.randn(N_TEST, K) * 1.5

main_logits = 1.0 * svc_logits + 0.40 * qwen14_logits + 0.20 * qwen25_logits + 0.25 * deb_logits
_main_raw = softmax(main_logits)

# 2. Comp stack logits (12 models) with golden V24 weight 0.15
comp_logits = np.random.randn(N_TEST, K) * 1.6
_comp = softmax(comp_logits)

P = 0.85 * _main_raw + 0.15 * _comp
print(f"P after Comp Blend: finite={np.isfinite(P).all()}, sum={np.allclose(P.sum(1), 1)}")

# 3. Sibling-Gated ATLAS Precision Focus
_sorted_idx = np.argsort(P, axis=1)
_top1 = _sorted_idx[:, -1]
_top2 = _sorted_idx[:, -2]
_margins = P[np.arange(N_TEST), _top1] - P[np.arange(N_TEST), _top2]
_same_fam = np.array([FAMS[_top1[i]] == FAMS[_top2[i]] for i in range(N_TEST)])

_atlas_rows = np.flatnonzero((_margins < 0.30) & _same_fam)
print(f"ATLAS Sibling-Gated active rows: {len(_atlas_rows)} / {N_TEST} ({len(_atlas_rows)/N_TEST*100:.1f}%)")

if len(_atlas_rows):
    atlas_prob = softmax(np.random.randn(len(_atlas_rows), K) * 2.0)
    P[_atlas_rows] = 0.90 * P[_atlas_rows] + 0.10 * atlas_prob
print(f"P after ATLAS: finite={np.isfinite(P).all()}, sum={np.allclose(P.sum(1), 1)}")

# 4. Night24 Spectral 4,404 Style Heads: 0.90 * P + 0.10 * hp
hp = softmax(np.random.randn(N_TEST, K) * 1.5)
P = 0.90 * P + 0.10 * hp
print(f"P after Spectral: finite={np.isfinite(P).all()}, sum={np.allclose(P.sum(1), 1)}")

# 5. Intra-Family Length Calibration (Mass-Invariant)
idx_g37 = LABELS.index('gemini-3.7-flash')
idx_g38 = LABELS.index('gemini-3.8-flash')
idx_glm = LABELS.index('glm-5.3')
idx_glmf = LABELS.index('glm-5.3-flash')

# Synthetic text lengths and markers
mock_texts = [
    "Short text with only a few words here. " * 3,               # ~21 words
    "Medium explanation passage with reasoning Step 1: and First, " * 10, # ~90 words
    "A very long and elaborate philosophical treatise discussing deep metaphysics and cosmic evolution " * 15, # ~180 words
    "这是一段包含「中文引号」的短文本。"
] * (N_TEST // 4 + 1)
mock_texts = mock_texts[:N_TEST]

_len_calib_count = 0
_gemini_sum_before = (P[:, idx_g37] + P[:, idx_g38]).copy()

for _i, _txt in enumerate(mock_texts):
    _wcount = len(_txt.split())
    if P[_i, idx_g38] + P[_i, idx_g37] > 0.08:
        if _wcount < 85:
            _delta = 0.06 * P[_i, idx_g37]
            P[_i, idx_g38] += _delta
            P[_i, idx_g37] -= _delta
            _len_calib_count += 1
        elif _wcount > 135:
            _delta = 0.06 * P[_i, idx_g38]
            P[_i, idx_g37] += _delta
            P[_i, idx_g38] -= _delta
            _len_calib_count += 1

_gemini_sum_after = P[:, idx_g37] + P[:, idx_g38]
assert np.allclose(_gemini_sum_before, _gemini_sum_after, atol=1e-7), "Gemini family mass invariance violated!"
print(f"Intra-Family Length Calibration applied to {_len_calib_count} samples. Mass invariance VERIFIED 100%!")

# 6. Sibling Micro-Gates
_g38_boost = 0
_glm_boost = 0
for _i, _txt in enumerate(mock_texts):
    if any(m in _txt for m in ['First,', 'Second,', 'Step 1:', '1. ', 'In conclusion,']):
        if P[_i, idx_g38] > 0.04:
            P[_i, idx_g38] *= 1.12
            P[_i, idx_g37] *= 0.90
            _g38_boost += 1
    if any(m in _txt for m in ['，', '。', '「', '」', '、', '《', '》']):
        if P[_i, idx_glm] > 0.04:
            P[_i, idx_glm] *= 1.15
            P[_i, idx_glmf] *= 0.88
            _glm_boost += 1

P = P / P.sum(1, keepdims=True)
print(f"P after Micro-Gates: finite={np.isfinite(P).all()}, sum={np.allclose(P.sum(1), 1)}")

# 7. Terminal Sinkhorn Purity (300 iters)
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
print("\nV26 SIMULATION TEST: ALL CHECKS PASSED 100% PERFECTLY!")
