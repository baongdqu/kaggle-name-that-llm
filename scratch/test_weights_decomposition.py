import numpy as np, pandas as pd
from scipy.special import logsumexp, softmax
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score

print("Loading test_cross_72_milestone components...", flush=True)
import test_cross_72_milestone as m

log_dl = m.log_dl
oof_plain = m.oof_svc_plain
oof_cent = m.oof_svc_centered
sc_opus = m.sc_opus
opus_idx = m.opus_idx

y_true = m.y_true
all_pids = m.all_pids
unique_pids = m.unique_pids
LABELS = m.LABELS
pid_indices = [np.where(all_pids == p)[0] for p in unique_pids]

data = np.load('scratch/cache_cv_data.npz', allow_pickle=True)
gem_cols = data['gemini_cols']
gem_probs = data['gemini_probs']
gem_rel = float(data['gemini_rel'])

gpt_cols = data['gpt56_cols']
gpt_probs = data['gpt56_probs']
gpt_rel = float(data['gpt56_rel'])

glm_cols = data['glm_cols']
glm_probs = data['glm_probs']
glm_rel = float(data['glm_rel'])

oof_fbl_glm = data['oof_fbl_glm_diff']
oof_gem_pair = data['oof_gem_pair_diff']
oof_st = data['oof_st_diff']

L2I = {l: i for i, l in enumerate(LABELS)}
idx_fbl = L2I['claude-fable-5']
idx_glm = L2I['glm-5.3']
idx_g37 = L2I['gemini-3.7-flash']
idx_g38 = L2I['gemini-3.8-flash']
idx_sol = L2I['gpt-5.6-sol']
idx_ter = L2I['gpt-5.6-terra']

def eval_logits(logits):
    preds = np.zeros(len(y_true), dtype=int)
    for idx in pid_indices:
        M = logits[idx]
        r, c = linear_sum_assignment(-M)
        preds[idx[r]] = c
    return balanced_accuracy_score(y_true, preds)

print("Testing different weights for Deep Learning vs Classical ML vs Opus Gate...", flush=True)

best_score = 0.7266
best_w = None

for w_dl in [0.8, 1.0, 1.2]:
    for w_p in [1.0, 1.5, 2.0]:
        for w_c in [1.0, 1.5, 2.0]:
            for w_op in [0.25, 0.35, 0.45]:
                base = w_dl * log_dl + w_p * oof_plain + w_c * oof_cent
                base[:, opus_idx] += w_op * sc_opus
                
                out = base.copy()
                
                # Apply champion residual scaling
                old_g = out[:, gem_cols]
                old_gp = softmax(old_g, axis=1)
                part_g = np.partition(old_gp, -2, axis=1)
                gate_g = np.clip((1.0 - (part_g[:, -1] - part_g[:, -2])), 0.0, 1.0)
                str_g = (0.9 * gem_rel) * gate_g
                mix_g = (1.0 - str_g[:, None]) * old_gp + str_g[:, None] * np.maximum(gem_probs, 1e-8)
                out[:, gem_cols] = logsumexp(old_g, axis=1, keepdims=True) + np.log(np.maximum(mix_g, 1e-12))
                
                old_p56 = out[:, gpt_cols]
                old_pp = softmax(old_p56, axis=1)
                part_p = np.partition(old_pp, -2, axis=1)
                gate_p = np.clip((1.0 - (part_p[:, -1] - part_p[:, -2])), 0.0, 1.0)
                str_p = (1.2 * gpt_rel) * gate_p
                mix_p = (1.0 - str_p[:, None]) * old_pp + str_p[:, None] * np.maximum(gpt_probs, 1e-8)
                out[:, gpt_cols] = logsumexp(old_p56, axis=1, keepdims=True) + np.log(np.maximum(mix_p, 1e-12))
                
                old_glm = out[:, glm_cols]
                old_glmp = softmax(old_glm, axis=1)
                part_glm = np.partition(old_glmp, -2, axis=1)
                gate_glm = np.clip((1.0 - (part_glm[:, -1] - part_glm[:, -2])), 0.0, 1.0)
                str_glm = (22.0 * glm_rel) * gate_glm
                mix_glm = (1.0 - str_glm[:, None]) * old_glmp + str_glm[:, None] * np.maximum(glm_probs, 1e-8)
                out[:, glm_cols] = logsumexp(old_glm, axis=1, keepdims=True) + np.log(np.maximum(mix_glm, 1e-12))
                
                out[:, idx_glm] += 0.35 * oof_fbl_glm
                out[:, idx_fbl] -= 0.35 * oof_fbl_glm
                
                out[:, idx_g38] += 0.15 * oof_gem_pair
                out[:, idx_g37] -= 0.15 * oof_gem_pair
                
                p_curr = softmax(out, axis=1)
                p_st = p_curr[:, idx_sol] + p_curr[:, idx_ter]
                out[:, idx_ter] += (0.08 * p_st) * oof_st
                out[:, idx_sol] -= (0.08 * p_st) * oof_st
                
                sc = eval_logits(out)
                if sc > best_score:
                    best_score = sc
                    best_w = (w_dl, w_p, w_c, w_op)
                    print(f"🔥 NEW BEST BALANCED ACCURACY: {best_score*100:.2f}% | w_dl={w_dl}, w_p={w_p}, w_c={w_c}, w_op={w_op}", flush=True)

print(f"\n🏆 PEAK BALANCED ACCURACY: {best_score*100:.2f}% with {best_w}")

