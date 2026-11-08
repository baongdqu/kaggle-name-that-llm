import numpy as np, pandas as pd
from scipy.special import logsumexp, softmax
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
import test_cross_72_milestone as m

base_logits = m.base_logits
cluster_oof = m.cluster_oof
oof_fbl_glm_diff = m.oof_fbl_glm_diff
oof_gem_pair_diff = m.oof_gem_pair_diff
idx_glm = m.idx_glm
idx_fbl = m.idx_fbl
idx_g37 = m.idx_g37
idx_g38 = m.idx_g38
y_true = m.y_true
unique_pids = m.unique_pids
all_pids = m.all_pids

def eval_raw_hungarian(logits):
    preds = np.zeros(len(y_true), dtype=int)
    for p in unique_pids:
        idx = np.where(all_pids == p)[0]
        M = logits[idx]
        r, c = linear_sum_assignment(-M)
        preds[idx[r]] = c
    return balanced_accuracy_score(y_true, preds), preds

print("🔍 Grid Searching with Direct Bipartite Maximum Weight Matching...", flush=True)

best_score = 0
best_params = None
best_preds = None

for a_glm in [8.0, 10.0, 12.0, 14.0, 16.0]:
    for a_gpt in [1.0, 1.2, 1.5]:
        for a_gem in [0.7, 0.9, 1.1]:
            for w_fg in [0.35, 0.40, 0.45, 0.50]:
                for w_gem in [0.10, 0.15, 0.20]:
                    out = base_logits.copy()
                    weights = {'gemini_flash': a_gem, 'gpt56': a_gpt, 'glm': a_glm}
                    
                    for fname in ['gemini_flash', 'gpt56', 'glm']:
                        cols, oof_probs, reliability = cluster_oof[fname]
                        alpha = weights[fname]
                        
                        old_logits = out[:, cols].copy()
                        old_p = softmax(old_logits, axis=1)
                        part = np.partition(old_p, -2, axis=1)
                        margin = part[:, -1] - part[:, -2]
                        gate = np.clip((1.0 - margin) / 1.0, 0.0, 1.0)
                        
                        strength = (alpha * reliability) * gate
                        style_p = np.maximum(oof_probs, 1e-8)
                        mix = (1.0 - strength[:, None]) * old_p + strength[:, None] * style_p
                        out[:, cols] = logsumexp(old_logits, axis=1, keepdims=True) + np.log(np.maximum(mix, 1e-12))
                    
                    out[:, idx_glm] += w_fg * oof_fbl_glm_diff
                    out[:, idx_fbl] -= w_fg * oof_fbl_glm_diff
                    
                    out[:, idx_g38] += w_gem * oof_gem_pair_diff
                    out[:, idx_g37] -= w_gem * oof_gem_pair_diff
                    
                    sc, preds = eval_raw_hungarian(out)
                    if sc > best_score:
                        best_score = sc
                        best_params = (a_glm, a_gpt, a_gem, w_fg, w_gem)
                        best_preds = preds
                        print(f"🔥 NEW BEST CV: {best_score*100:.2f}% | glm={a_glm}, gpt={a_gpt}, gem={a_gem}, fg={w_fg}, gem_pair={w_gem}", flush=True)

print(f"\n==========================================")
print(f"🏆 CHAMPION BALANCED ACCURACY: {best_score*100:.2f}%")
print(f"Optimal Parameters: {best_params}")
print(f"==========================================")

LABELS = m.LABELS
for i, l in enumerate(LABELS):
    mask = (y_true == i)
    acc = (best_preds[mask] == i).mean()
    cnt = (best_preds[mask] == i).sum()
    tot = mask.sum()
    print(f"  {l:25s}: {acc*100:5.2f}% ({cnt:3d} / {tot:3d})")

