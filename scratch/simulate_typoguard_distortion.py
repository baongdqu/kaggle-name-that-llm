import numpy as np
import pandas as pd

LABELS = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]
K = len(LABELS)

train = pd.read_csv('data/train.csv')
N = len(train)

# Realistic noisy probabilities centered around true labels (~0.64 accuracy)
np.random.seed(42)
true_indices = train.model.map({l: i for i, l in enumerate(LABELS)}).values

base_logits = np.random.randn(N, K) * 1.0
for i, t_idx in enumerate(true_indices):
    base_logits[i, t_idx] += 1.45  # calibrated to give ~0.64 balanced accuracy

P_clean = np.exp(base_logits - base_logits.max(axis=1, keepdims=True))
P_clean /= P_clean.sum(axis=1, keepdims=True)

def sinkhorn(P, iters=300):
    c = np.ones(K)
    target = np.full(K, len(P) / K)
    for _ in range(iters):
        R = P * c
        R /= R.sum(1, keepdims=True)
        c *= target / R.sum(0)
    R = P * c
    return R / R.sum(1, keepdims=True)

P_base_sink = sinkhorn(P_clean.copy())
acc_base = (P_base_sink.argmax(axis=1) == true_indices).mean()
print(f"Baseline Accuracy with Sinkhorn: {acc_base:.4f}")

# Now apply V23 Soft TypoGuard
P_typo = P_clean.copy()
idx_claude_glm = [i for i, l in enumerate(LABELS) if l.startswith(('claude', 'glm'))]
idx_gemini = [i for i, l in enumerate(LABELS) if l.startswith('gemini')]
idx_gpt_straight = [i for i, l in enumerate(LABELS) if l in {'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra'}]

for _i, _txt in enumerate(train.text.values):
    if '’' in _txt:
        P_typo[_i, idx_claude_glm] *= 0.15
    if '—' in _txt:
        P_typo[_i, idx_gemini] *= 0.15
    if '"' in _txt and not ('“' in _txt or '”' in _txt):
        P_typo[_i, idx_gpt_straight] *= 0.20

P_typo /= P_typo.sum(axis=1, keepdims=True)
P_typo_sink = sinkhorn(P_typo)
acc_typo = (P_typo_sink.argmax(axis=1) == true_indices).mean()
print(f"With Soft TypoGuard + Sinkhorn on Train: {acc_typo:.4f}")

# NOW: What happens if on the HIDDEN TEST SET, punctuation distribution shifts slightly?
# E.g., Gemini has just 5% em-dashes on test set, or Claude has 5% curly apostrophes?
P_test_sim = P_clean.copy()
# Suppose in 5% of true Gemini rows, there is an em-dash, and in 5% of true Claude rows, there is a curly apostrophe
for _i, _txt in enumerate(train.text.values):
    t_idx = true_indices[_i]
    is_gemini = LABELS[t_idx].startswith('gemini')
    is_claude = LABELS[t_idx].startswith('claude')
    # If test prompt induced curly apostrophe in 10% of Claude or em-dash in 10% of Gemini:
    has_curly = ('’' in _txt) or (is_claude and np.random.rand() < 0.10)
    has_emdash = ('—' in _txt) or (is_gemini and np.random.rand() < 0.10)
    if has_curly:
        P_test_sim[_i, idx_claude_glm] *= 0.15
    if has_emdash:
        P_test_sim[_i, idx_gemini] *= 0.15

P_test_sim /= P_test_sim.sum(axis=1, keepdims=True)
P_test_sink = sinkhorn(P_test_sim)
acc_test_shift = (P_test_sink.argmax(axis=1) == true_indices).mean()
print(f"With Shifted Test Distribution + TypoGuard + Sinkhorn: {acc_test_shift:.4f}")
print(f"Drop due to distribution violation: {acc_test_shift - acc_base:+.4f}")
