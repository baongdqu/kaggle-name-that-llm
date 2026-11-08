import numpy as np, time
from scipy.special import logsumexp, softmax
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score

t0 = time.time()
print("⚡ Loading precomputed cache...", flush=True)
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
oof_so = data['oof_so_diff']
oof_kg = data['oof_kg_diff']
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
idx_son = L2I['claude-sonnet-5']
idx_op8 = L2I['claude-opus-4-8']
idx_kim = L2I['kimi-k3']
idx_g55 = L2I['gpt-5.5']
idx_sol = L2I['gpt-5.6-sol']
idx_ter = L2I['gpt-5.6-terra']

print(f"Loaded cache in {time.time()-t0:.2f}s!", flush=True)

# Pre-index prompts for maximum speed
pid_indices = [np.where(all_pids == p)[0] for p in unique_pids]

def fast_eval_matching(logits):
    preds = np.zeros(len(y_true), dtype=int)
    for idx in pid_indices:
        M = logits[idx]
        r, c = linear_sum_assignment(-M)
        preds[idx[r]] = c
    return balanced_accuracy_score(y_true, preds), preds

print("🔍 Running Comprehensive Fast Grid Search...", flush=True)

best_score = 0
best_params = None
best_preds = None

for a_glm in [14.0, 16.0, 18.0, 20.0, 22.0]:
    for a_gpt in [1.0, 1.2, 1.4]:
        for a_gem in [0.8, 0.9, 1.0]:
            out = base_logits.copy()
            
            # 1. Gemini Flash residual
            old_g = out[:, gem_cols]
            old_gp = softmax(old_g, axis=1)
            part_g = np.partition(old_gp, -2, axis=1)
            gate_g = np.clip((1.0 - (part_g[:, -1] - part_g[:, -2])), 0.0, 1.0)
            str_g = (a_gem * gem_rel) * gate_g
            mix_g = (1.0 - str_g[:, None]) * old_gp + str_g[:, None] * np.maximum(gem_probs, 1e-8)
            out[:, gem_cols] = logsumexp(old_g, axis=1, keepdims=True) + np.log(np.maximum(mix_g, 1e-12))
            
            # 2. GPT-5.6 residual
            old_p56 = out[:, gpt_cols]
            old_pp = softmax(old_p56, axis=1)
            part_p = np.partition(old_pp, -2, axis=1)
            gate_p = np.clip((1.0 - (part_p[:, -1] - part_p[:, -2])), 0.0, 1.0)
            str_p = (a_gpt * gpt_rel) * gate_p
            mix_p = (1.0 - str_p[:, None]) * old_pp + str_p[:, None] * np.maximum(gpt_probs, 1e-8)
            out[:, gpt_cols] = logsumexp(old_p56, axis=1, keepdims=True) + np.log(np.maximum(mix_p, 1e-12))
            
            # 3. GLM residual
            old_glm = out[:, glm_cols]
            old_glmp = softmax(old_glm, axis=1)
            part_glm = np.partition(old_glmp, -2, axis=1)
            gate_glm = np.clip((1.0 - (part_glm[:, -1] - part_glm[:, -2])), 0.0, 1.0)
            str_glm = (a_glm * glm_rel) * gate_glm
            mix_glm = (1.0 - str_glm[:, None]) * old_glmp + str_glm[:, None] * np.maximum(glm_probs, 1e-8)
            out[:, glm_cols] = logsumexp(old_glm, axis=1, keepdims=True) + np.log(np.maximum(mix_glm, 1e-12))
            
            p_curr = softmax(out, axis=1)
            p_st = p_curr[:, idx_sol] + p_curr[:, idx_ter]
            p_so = p_curr[:, idx_son] + p_curr[:, idx_op8]
            
            for w_fg in [0.30, 0.35, 0.40]:
                for w_gem in [0.12, 0.15, 0.18]:
                    for w_st in [0.0, 0.08, 0.12, 0.16]:
                        temp_out = out.copy()
                        temp_out[:, idx_glm] += w_fg * oof_fbl_glm
                        temp_out[:, idx_fbl] -= w_fg * oof_fbl_glm
                        
                        temp_out[:, idx_g38] += w_gem * oof_gem_pair
                        temp_out[:, idx_g37] -= w_gem * oof_gem_pair
                        
                        if w_st > 0:
                            temp_out[:, idx_ter] += (w_st * p_st) * oof_st
                            temp_out[:, idx_sol] -= (w_st * p_st) * oof_st
                            
                        sc, preds = fast_eval_matching(temp_out)
                        if sc > best_score:
                            best_score = sc
                            best_params = (a_glm, a_gpt, a_gem, w_fg, w_gem, w_st)
                            best_preds = preds
                            print(f"🔥 NEW BEST BALANCED ACCURACY: {best_score*100:.2f}% | glm={a_glm}, gpt={a_gpt}, gem={a_gem}, fg={w_fg}, gem_p={w_gem}, st={w_st}", flush=True)

print(f"\n==========================================")
print(f"🏆 CHAMPION BALANCED ACCURACY: {best_score*100:.2f}%")
print(f"Optimal Parameters: {best_params}")
print(f"==========================================")

for i, l in enumerate(LABELS):
    mask = (y_true == i)
    acc = (best_preds[mask] == i).mean()
    cnt = (best_preds[mask] == i).sum()
    tot = mask.sum()
    print(f"  {l:25s}: {acc*100:5.2f}% ({cnt:3d} / {tot:3d})")