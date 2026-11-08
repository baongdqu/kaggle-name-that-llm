import numpy as np
import pandas as pd

LABELS = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]
K = len(LABELS)

def get_family(label):
    for f in ['gpt', 'gemini', 'claude', 'glm', 'qwen', 'kimi']:
        if label.startswith(f):
            return f
    return 'other'

LABEL_FAMILIES = [get_family(l) for l in LABELS]

N_TEST = 4000
np.random.seed(42)

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

raw_logits = np.random.randn(N_TEST, K) * 1.8
P = softmax(raw_logits)

# Margins
sorted_indices = np.argsort(P, axis=1)
top1_idx = sorted_indices[:, -1]
top2_idx = sorted_indices[:, -2]
margins = P[np.arange(N_TEST), top1_idx] - P[np.arange(N_TEST), top2_idx]

# Check how many are intra-family ambiguous
same_fam = np.array([LABEL_FAMILIES[top1_idx[i]] == LABEL_FAMILIES[top2_idx[i]] for i in range(N_TEST)])
margin_mask = margins < 0.30

both_mask = margin_mask & same_fam
diff_fam_mask = margin_mask & (~same_fam)

print(f"Total test rows: {N_TEST}")
print(f"Rows with margin < 0.30: {margin_mask.sum()} ({margin_mask.sum()/N_TEST*100:.1f}%)")
print(f"Rows with margin < 0.30 AND SAME FAMILY (Sibling Ambiguity): {both_mask.sum()} ({both_mask.sum()/N_TEST*100:.1f}%)")
print(f"Rows with margin < 0.30 BUT DIFFERENT FAMILY: {diff_fam_mask.sum()} ({diff_fam_mask.sum()/N_TEST*100:.1f}%)")
