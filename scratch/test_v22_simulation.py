import numpy as np
import pandas as pd
import time

LABELS = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]
K = len(LABELS)
N = 4000
np.random.seed(42)

print(f"=== TESTING CHAMPION V22 PRE-SINKHORN SOVEREIGN SIMULATION (N={N}) ===")

# 1. Generate realistic synthetic test texts
sample_texts = [
    "First, we analyze the prompt. Step 1: break down the logic. In conclusion, this is clear.",
    "This is a general passage discussing artificial intelligence and algorithmic models.",
    "这里有一些中文標點符號：例如「引號」以及句號。整體表現非常穩定。",
    "Brief explanation without extra markers or unusual punctuation.",
    "Step 1: let's calculate the derivation. Second, verify the balance.",
    "普通文字，包含，逗号与句号。"
]
test_texts = [sample_texts[i % len(sample_texts)] for i in range(N)]
test = pd.DataFrame({"id": [f"id_{i:04d}" for i in range(N)], "text": test_texts})

# 2. Simulate raw ensemble probabilities P (shape: 4000, 18)
logits = np.random.randn(N, K).astype(np.float32)
exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
P = exp_logits / exp_logits.sum(axis=1, keepdims=True)

assert P.shape == (N, K)
assert np.allclose(P.sum(axis=1), 1.0)
print("1. Raw ensemble probabilities initialized successfully.")

# 3. Apply Pre-Sinkhorn Sibling Micro-Gates
idx_g37 = LABELS.index('gemini-3.7-flash')
idx_g38 = LABELS.index('gemini-3.8-flash')
idx_glm = LABELS.index('glm-5.3')
idx_glmf = LABELS.index('glm-5.3-flash')

_g38_boost_count = 0
_glm_boost_count = 0

t0 = time.time()
for _i, _txt in enumerate(test.text.values):
    # Gemini reasoning prefix gate
    if any(m in _txt for m in ['First,', 'Second,', 'Step 1:', '1. ', 'In conclusion,']):
        if P[_i, idx_g38] > 0.04:
            P[_i, idx_g38] *= 1.12
            P[_i, idx_g37] *= 0.90
            _g38_boost_count += 1
    # GLM Chinese Unicode marks gate
    if any(m in _txt for m in ['，', '。', '「', '」', '、', '《', '》']):
        if P[_i, idx_glm] > 0.04:
            P[_i, idx_glm] *= 1.15
            P[_i, idx_glmf] *= 0.88
            _glm_boost_count += 1

# Renormalize rows before Sinkhorn
P = P / P.sum(1, keepdims=True)
assert np.isfinite(P).all() and np.allclose(P.sum(1), 1.0)
print(f"2. Pre-Sinkhorn Micro-Gates applied: {_g38_boost_count} Gemini reasoning boosts, {_glm_boost_count} GLM Unicode boosts.")

# 4. Terminal Sinkhorn Balancing
def sinkhorn(P, iters=300):
    c = np.ones(K)
    target = np.full(K, len(P) / K)
    for _ in range(iters):
        R = P * c
        R /= R.sum(1, keepdims=True)
        c *= target / R.sum(0)
    R = P * c
    return R / R.sum(1, keepdims=True)

P = sinkhorn(P, iters=300)
assert np.isfinite(P).all() and np.allclose(P.sum(1), 1.0)

# Check column sums
col_sums = P.sum(axis=0)
expected_col_sum = N / K
max_col_err = np.abs(col_sums - expected_col_sum).max()
print(f"3. Terminal Sinkhorn converged: max column sum error = {max_col_err:.8f} (Target: {expected_col_sum:.2f} per class)")
assert max_col_err < 1e-4, f"Column sum error too large: {max_col_err}"

# 5. Argmax prediction and distribution check
_final_indices = P.argmax(axis=1)
assert len(_final_indices) == N
sub = pd.DataFrame({"id": test.id.astype(str), "model": np.array(LABELS)[_final_indices]})
assert sub.id.is_unique and sub.model.isin(LABELS).all()

# Class distribution
counts = sub["model"].value_counts()
print(f"4. Class distribution min: {counts.min()}, max: {counts.max()}, mean: {counts.mean():.1f}")
print(f"5. Total execution time of V22 post-processing: {(time.time() - t0)*1000:.2f} ms")

sub.to_parquet("scratch/test_v22_sub.parquet", index=False)
print("=== ALL TESTS PASSED 100% FOR CHAMPION V22 PRE-SINKHORN SOVEREIGN! ===")
