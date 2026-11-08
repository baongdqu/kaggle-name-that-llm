import numpy as np

LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3', 'glm-5.3-flash',
          'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']
K = len(LABELS)

# Class-adaptive margin dictionary
# Default: margin >= 0.30, top_p >= 0.48
# Rare / distinct signature classes: margin >= 0.24, top_p >= 0.42
# Dense siblings: margin >= 0.35, top_p >= 0.50
DELTA_THRESHOLDS = np.full(K, 0.30, dtype=np.float32)
P_THRESHOLDS = np.full(K, 0.48, dtype=np.float32)

HIGH_SENSITIVITY = ['gpt-5.5', 'kimi-k3', 'gemini-3.1-pro-preview', 'claude-fable-5', 'glm-5.3-flash']
DENSE_CLUSTERS = ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'claude-opus-4-6', 'claude-opus-5']

for name in HIGH_SENSITIVITY:
    idx = LABELS.index(name)
    DELTA_THRESHOLDS[idx] = 0.24
    P_THRESHOLDS[idx] = 0.42

for name in DENSE_CLUSTERS:
    idx = LABELS.index(name)
    DELTA_THRESHOLDS[idx] = 0.34
    P_THRESHOLDS[idx] = 0.50

print("Delta thresholds:", DELTA_THRESHOLDS)
print("P thresholds:", P_THRESHOLDS)
print("Adaptive Anchor configuration validated successfully!")

