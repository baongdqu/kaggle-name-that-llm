import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp, softmax

data = np.load('scratch/cache_cv_data.npz', allow_pickle=True)
base_logits = data['base_logits']
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

y_true = data['y_true']
all_pids = data['all_pids']
unique_pids = data['unique_pids']
LABELS = list(data['labels'])
L2I = {l: i for i, l in enumerate(LABELS)}

idx_fbl = L2I['claude-fable-5']
idx_glm = L2I['glm-5.3']
idx_g37 = L2I['gemini-3.7-flash']
idx_g38 = L2I['gemini-3.8-flash']
idx_sol = L2I['gpt-5.6-sol']
idx_ter = L2I['gpt-5.6-terra']

pid_indices = [np.where(all_pids == p)[0] for p in unique_pids]

out = base_logits.copy()

# Apply optimal parameters: glm=22.0, gpt=1.2, gem=0.9
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

preds = np.zeros(len(y_true), dtype=int)
for idx in pid_indices:
    M = out[idx]
    r, c = linear_sum_assignment(-M)
    preds[idx[r]] = c

cm = confusion_matrix(y_true, preds)
confused = []
for i in range(18):
    for j in range(18):
        if i != j and cm[i, j] > 0:
            confused.append((cm[i, j], LABELS[i], LABELS[j]))

confused.sort(reverse=True)
print("\n=== TOP 20 CONFUSED PAIRS AT 72.66% / 72.73% ===")
for cnt, t, p in confused[:20]:
    print(f"{cnt:3d} errors: True [{t:22s}] -> Predicted as [{p:22s}]")

