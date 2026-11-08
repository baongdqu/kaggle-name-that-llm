import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
import test_cross_72_milestone as m

base_out = m.out # The best logits from 71.91%
y_true = m.y_true
unique_pids = m.unique_pids
all_pids = m.all_pids

print("--- Testing Matching Algorithms & Scaling ---", flush=True)

# 1. Pure Linear Sum Assignment (Raw Hungarian on -Logits)
preds_raw = np.zeros(len(y_true), dtype=int)
for p in unique_pids:
    idx = np.where(all_pids == p)[0]
    M = base_out[idx]
    r, c = linear_sum_assignment(-M)
    preds_raw[idx[r]] = c
bacc_raw = balanced_accuracy_score(y_true, preds_raw)
print(f"Raw Hungarian (-Logits): {bacc_raw*100:.2f}%")

# 2. Sinkhorn at various temperatures and iterations
for iters in [5, 10, 15, 20, 30]:
    for temp in [0.20, 0.25, 0.30, 0.35, 0.38, 0.40, 0.45, 0.50]:
        preds = np.zeros(len(y_true), dtype=int)
        for p in unique_pids:
            idx = np.where(all_pids == p)[0]
            M = base_out[idx]
            K = np.exp(M / temp)
            for _ in range(iters):
                K = K / (K.sum(axis=1, keepdims=True) + 1e-12)
                K = K / (K.sum(axis=0, keepdims=True) + 1e-12)
            r, c = linear_sum_assignment(-K)
            preds[idx[r]] = c
        bacc = balanced_accuracy_score(y_true, preds)
        if bacc >= 0.7190:
            print(f"Sinkhorn iters={iters:2d}, temp={temp:.2f}: {bacc*100:.2f}%")

