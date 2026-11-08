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

print("=== STEP 1: VERIFY CURATED POOL PARQUET ===")
t0 = time.time()
pool_path = "scratch/curated_pool/curated_pool.parquet"
pool = pd.read_parquet(pool_path)
print(f"Loaded {pool_path}: shape {pool.shape} in {time.time()-t0:.2f}s")
assert "register_ok" in pool.columns
assert "model" in pool.columns
assert "text" in pool.columns

pool_clean = pool[pool["register_ok"] == True].copy()
print(f"Register OK certified passages: {len(pool_clean)} / {len(pool)}")
assert set(pool_clean["model"].unique()) == set(LABELS), "Mismatch in 18 model classes!"

# Verify mapping
mapped_y = pool_clean["model"].map({l: i for i, l in enumerate(LABELS)}).values
assert not np.isnan(mapped_y).any(), "Found NaN in model mapping!"
print(f"Step 1 PASS: Curated pool ready with {len(pool_clean)} certified passages across all {K} classes.")

print("\n=== STEP 2: SIMULATE 4,000 TEST SAMPLES DECISION FLOW WITH TYPOGUARD & GATES ===")
N_TEST = 4000
np.random.seed(42)

# Synthetic test df with diverse typographic and sibling markers
mock_texts = [
    "It’s an extraordinary day for quantum physics and celestial mechanics.",          # Curly apostrophe -> Claude/GLM suppressed
    "The journey was arduous — yet illuminating for every astronomer present.",         # Em-dash -> Gemini suppressed
    'He said "truth is stranger than fiction" without flinching.',                     # Straight quotes -> GPT-5.6/6 suppressed
    "First, we analyze Step 1: calculate the derivative. In conclusion, the answer is 42.", # Gemini reasoning prefix
    "这是一段测试中文，使用了标点符号「引号」以及句号。",                                     # GLM Chinese Unicode
    "Plain standard English sentence with no special punctuation whatsoever."
] * (N_TEST // 6 + 1)
mock_texts = mock_texts[:N_TEST]

test_df = pd.DataFrame({
    "id": [f"test_{i:04d}" for i in range(N_TEST)],
    "text": mock_texts
})

# Mock probability matrix P (from ensemble 85% + comp 15% + atlas 10% + spectral 10%)
raw_logits = np.random.randn(N_TEST, K) * 1.5
P = np.exp(raw_logits - raw_logits.max(axis=1, keepdims=True))
P = P / P.sum(axis=1, keepdims=True)

# Pre-Sinkhorn Sibling Micro-Gates & Soft TypoGuard
idx_g37 = LABELS.index('gemini-3.7-flash')
idx_g38 = LABELS.index('gemini-3.8-flash')
idx_glm = LABELS.index('glm-5.3')
idx_glmf = LABELS.index('glm-5.3-flash')
idx_claude_glm = [i for i, l in enumerate(LABELS) if l.startswith(('claude', 'glm'))]
idx_gemini = [i for i, l in enumerate(LABELS) if l.startswith('gemini')]
idx_gpt_straight = [i for i, l in enumerate(LABELS) if l in {'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra'}]

_g38_boost_count = 0
_glm_boost_count = 0
_typo_cg_count = 0
_typo_gem_count = 0
_typo_gpt_count = 0

for _i, _txt in enumerate(test_df["text"].values):
    # 1. Soft Pre-Sinkhorn TypoGuard (Safe orthographic suppression)
    if '’' in _txt:
        P[_i, idx_claude_glm] *= 0.15
        _typo_cg_count += 1
    if '—' in _txt:
        P[_i, idx_gemini] *= 0.15
        _typo_gem_count += 1
    if '"' in _txt and not ('“' in _txt or '”' in _txt):
        P[_i, idx_gpt_straight] *= 0.20
        _typo_gpt_count += 1

    # 2. Sibling micro-gates
    if any(m in _txt for m in ['First,', 'Second,', 'Step 1:', '1. ', 'In conclusion,']):
        if P[_i, idx_g38] > 0.04:
            P[_i, idx_g38] *= 1.12
            P[_i, idx_g37] *= 0.90
            _g38_boost_count += 1
    if any(m in _txt for m in ['，', '。', '「', '」', '、', '《', '》']):
        if P[_i, idx_glm] > 0.04:
            P[_i, idx_glm] *= 1.15
            P[_i, idx_glmf] *= 0.88
            _glm_boost_count += 1

P = P / P.sum(axis=1, keepdims=True)
assert np.isfinite(P).all() and np.allclose(P.sum(1), 1)
print(f"Applied TypoGuard: {_typo_cg_count} curly-apostrophe Claude/GLM cuts, {_typo_gem_count} em-dash Gemini cuts, {_typo_gpt_count} straight-quote GPT cuts.")
print(f"Applied Micro-Gates: {_g38_boost_count} Gemini reasoning boosts, {_glm_boost_count} GLM Unicode boosts.")

# Terminal Sinkhorn-Knopp (300 iters)
def sinkhorn(P, iters=300):
    c = np.ones(K)
    target = np.full(K, len(P) / K)
    for _ in range(iters):
        R = P * c
        R /= R.sum(1, keepdims=True)
        c *= target / R.sum(0)
    R = P * c
    return R / R.sum(1, keepdims=True)

t_sink = time.time()
P_balanced = sinkhorn(P, iters=300)
sink_time = time.time() - t_sink

print(f"Sinkhorn converged in {sink_time*1000:.2f} ms")
assert np.isfinite(P_balanced).all() and np.allclose(P_balanced.sum(1), 1)

# Check column sums
col_sums = P_balanced.sum(axis=0)
target_sum = N_TEST / K
max_col_err = np.max(np.abs(col_sums - target_sum))
print(f"Target column sum: {target_sum:.4f}")
print(f"Max column marginal deviation: {max_col_err:.8f}")
assert max_col_err < 1e-4, f"Column marginal error too high: {max_col_err}"

# argmax prediction
final_indices = P_balanced.argmax(axis=1)
preds = np.array(LABELS)[final_indices]
sub = pd.DataFrame({"id": test_df["id"], "model": preds})
assert len(sub) == N_TEST
assert sub["id"].is_unique
assert sub["model"].isin(LABELS).all()

print("\nPrediction distribution across 18 models:")
print(sub["model"].value_counts().to_string())

print("\n=== ALL V23 PERFECTED TESTS & SIMULATIONS PASSED 100%! ===")
