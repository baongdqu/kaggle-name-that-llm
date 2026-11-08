import numpy as np
import pandas as pd
from scipy.special import log_softmax

def topic_correction(core, svc, groups, alpha=0.62, weight=0.22):
    core = np.asarray(core, dtype=np.float64)
    groups = np.asarray(groups)
    if alpha == 0 or svc is None or weight == 0:
        return core.copy()
    scores = np.asarray(svc, dtype=np.float64)
    if scores.shape != core.shape or len(groups) != len(core):
        raise ValueError('SVC evidence axes differ')
    centered = scores.copy()
    _, inverse, count = np.unique(groups, return_inverse=True, return_counts=True)
    eligible = (count[inverse] >= 8) & (count[inverse] <= 18)
    for g in np.unique(groups[eligible]):
        ix = groups == g
        centered[ix] -= scores[ix].mean(0, keepdims=True)
    delta = log_softmax(centered, axis=1) - log_softmax(scores, axis=1)
    delta[~eligible] = 0
    ordered = np.sort(core, axis=1)
    gate = np.clip((2 - (ordered[:, -1] - ordered[:, -2])) / 1.6, 0.15, 1.0)
    gate[~eligible] = 0
    return core + float(alpha) * float(weight) * delta * gate[:, None]

# Test dummy shapes
N, K = 100, 18
dummy_core = np.random.randn(N, K)
dummy_svc = np.random.randn(N, K)
dummy_groups = np.repeat(np.arange(10), 10) # 10 groups of 10 samples

res = topic_correction(dummy_core, dummy_svc, dummy_groups, alpha=0.62, weight=0.22)
print("Input shape:", dummy_core.shape)
print("Output shape:", res.shape)
print("Has NaN:", np.isnan(res).any())
print("Max diff:", np.abs(res - dummy_core).max())
print("Test completed successfully!")
