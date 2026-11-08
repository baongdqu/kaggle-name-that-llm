import sys
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax

# 1. Load Data & Cache
train = pd.read_csv('data/train.csv')
LABELS = sorted(train.model.unique())
L2I = {l: i for i, l in enumerate(LABELS)}
y_true = np.array([L2I[m] for m in train.model.values])
prompts = train.prompt_id.values
p2i = {p: i for i, p in enumerate(np.unique(prompts))}
g_true = np.array([p2i[p] for p in prompts])
K = 18

cache = np.load('lab/cache/base_oof.npz')
base_probs = cache['probs'].copy()

# Features
wlens = np.array([len(str(t).split()) for t in train.text.values], dtype=np.float32)
char_lens = np.array([len(str(t)) for t in train.text.values], dtype=np.float32)
avg_wlens = char_lens / np.maximum(wlens, 1.0)

# 2. Get baseline V68 predictions to identify 1,538 core stubborn errors & 1,161 correct samples
sys.path.append('.')
sys.path.append('lab')
from error_overlap_diagnostic import stubborn_mask, preds

pred_v68_base = preds['V68_Orthogonal']
base_score = balanced_accuracy_score(y_true, pred_v68_base)

n_stubborn = stubborn_mask.sum()
correct_mask = ~stubborn_mask
n_correct = correct_mask.sum()

print("=================================================================")
print(f"UNANCHORING EXPERIMENT: STEP 1 - SINKHORN ABLATION")
print(f"Total dataset: {len(train)} samples")
print(f"Core Stubborn Errors (Target to Fix) : {n_stubborn} samples")
print(f"Clean Correct Baseline (Must Protect): {n_correct} samples")
print(f"Base V68 Balanced Accuracy: {base_score:.4f}")
print("=================================================================\n")

# Sibling Groups
TRIO_GEMINI = [6, 7, 8]   # gemini 3.6, 3.7, 3.8
TRIO_GPT = [12, 13, 14]   # gpt 5.6 luna, sol, terra
DUO_GLM = [9, 10]         # glm 5.3, 5.3-flash

LEN_BIAS = np.array([-0.0306, -0.0011, +0.1664, +0.1764, +0.0556, +0.0807,
                     -0.0362, +0.0387, -0.2429, -0.0667, +0.0950, -0.0619,
                     +0.0085, +0.0521, -0.0236, +0.0372, -0.1470, -0.1007], dtype=np.float32)
WLEN_BIAS = np.array([-0.1886, +0.3526, -0.0640, -1.1015, +0.5982, -0.0520,
                      +0.5204, +1.1509, +1.0350, -0.2390, -0.1549, -0.7586,
                      +0.1915, -0.0895, -0.1149, -0.6532, -0.3990, -0.0348], dtype=np.float32)

# 3. Parameterized Pipeline
def evaluate_sinkhorn_variation(sinkhorn_w, target_mix=0.35, use_hungarian=True):
    p = base_probs.copy()
    
    # V68 Relative Calibrations
    p_mod = p.copy()
    for g in np.unique(g_true):
        idx = np.where(g_true == g)[0]
        gsize = len(idx)
        if gsize >= 3:
            adj = np.zeros((gsize, 18), dtype=np.float32)
            l_sub = wlens[idx]; s_l = l_sub.std()
            if s_l > 1.0:
                zl = (l_sub - l_sub.mean()) / s_l
                adj += 0.20 * np.outer(zl, LEN_BIAS)
            if gsize >= 6:
                w_sub = avg_wlens[idx]; s_w = w_sub.std()
                if s_w > 0.05:
                    zw = (w_sub - w_sub.mean()) / s_w
                    adj += 0.05 * np.outer(zw, WLEN_BIAS)
            if np.any(adj != 0):
                sub_p = p_mod[idx] * np.exp(adj)
                p_mod[idx] = sub_p / (sub_p.sum(axis=1, keepdims=True) + 1e-12)
                
    # Soft dispersion
    for g in np.unique(g_true):
        idx = np.where(g_true == g)[0]
        if len(idx) >= 2:
            t1 = p_mod[idx].argmax(axis=1)
            for c in range(18):
                c_rows = idx[t1 == c]
                if len(c_rows) >= 2:
                    confs = p_mod[c_rows, c]
                    ord_c = np.argsort(-confs)
                    for loser in c_rows[ord_c[1:]]:
                        p_mod[loser, c] *= 0.65
                    p_mod[c_rows] /= (p_mod[c_rows].sum(axis=1, keepdims=True) + 1e-12)
    P = p_mod / p_mod.sum(axis=1, keepdims=True)
    
    # Sinkhorn
    if sinkhorn_w > 0:
        Q = P.copy()
        pred_marg = np.clip(P.mean(0), 1e-6, None); pred_marg /= pred_marg.sum()
        train_prior = np.bincount(y_true, minlength=18).astype(np.float64) + 1.0
        train_prior /= train_prior.sum()
        target = (1.0 - target_mix) * pred_marg + target_mix * train_prior
        target = np.clip(target, 1e-6, None); target /= target.sum()
        for _ in range(60):
            Q *= (len(Q) * target[None, :]) / np.clip(Q.sum(0, keepdims=True), 1e-12, None)
            Q /= np.clip(Q.sum(1, keepdims=True), 1e-12, None)
        tot = (1.0 - sinkhorn_w) * np.log(P + 1e-12) + sinkhorn_w * np.log(Q + 1e-12)
    else:
        tot = np.log(P + 1e-12)
        
    pred = tot.argmax(axis=1).copy()
    
    if use_hungarian:
        c_sizes = np.bincount(g_true)[g_true]
        dense = np.unique(g_true[(c_sizes >= 5) & (c_sizes <= 18)])
        probs_tot = softmax(tot, axis=1)
        for g in dense:
            i_g = np.where(g_true == g)[0]
            if len(i_g) <= 1: continue
            sub_p = probs_tot[i_g]; sub_log = tot[i_g]
            sort_p = np.sort(sub_p, axis=1)
            margins = sort_p[:, -1] - sort_p[:, -2]
            top_p = sort_p[:, -1]; top_c = sub_log.argmax(axis=1)
            
            is_anc = (margins >= 0.20) & (top_p >= 0.35)
            assigned = set(); anc_idx = []
            for l_idx in np.argsort(-top_p):
                if is_anc[l_idx]:
                    cls_ = top_c[l_idx]
                    if cls_ not in assigned:
                        assigned.add(cls_); anc_idx.append(l_idx)
            unassigned = [i for i in range(len(i_g)) if i not in anc_idx]
            if unassigned:
                avail = [c for c in range(K) if c not in assigned]
                if len(avail) >= len(unassigned):
                    cost = -sub_log[unassigned][:, avail]
                    r_ind, c_ind = linear_sum_assignment(cost)
                    for r, c in zip(r_ind, c_ind):
                        pred[i_g[unassigned[r]]] = avail[c]
                        
    # Evaluation Metrics
    score = balanced_accuracy_score(y_true, pred)
    
    # 1. Core Stubborn Error Recovery (Within 1,538 errors)
    recovered = np.sum((pred[stubborn_mask] == y_true[stubborn_mask]))
    
    # 2. Collateral Damage (Within 1,161 correct samples)
    damaged = np.sum((pred[correct_mask] != y_true[correct_mask]))
    
    # Net Gain & Ratio
    net_diff = recovered - damaged
    ratio = recovered / max(damaged, 1)
    
    # Sibling Triangles Breakdown
    gemini_mask = stubborn_mask & np.isin(y_true, TRIO_GEMINI)
    gemini_recovered = np.sum((pred[gemini_mask] == y_true[gemini_mask]))
    
    gpt_mask = stubborn_mask & np.isin(y_true, TRIO_GPT)
    gpt_recovered = np.sum((pred[gpt_mask] == y_true[gpt_mask]))
    
    glm_mask = stubborn_mask & np.isin(y_true, DUO_GLM)
    glm_recovered = np.sum((pred[glm_mask] == y_true[glm_mask]))
    
    return {
        'score': score,
        'recovered': recovered,
        'damaged': damaged,
        'net': net_diff,
        'ratio': ratio,
        'gemini_rec': gemini_recovered,
        'gpt_rec': gpt_recovered,
        'glm_rec': glm_recovered
    }

# Run Damped Sinkhorn Grid from 0.45 down to 0.00
print(f"{'Sinkhorn':10s} | {'Hungarian':9s} | {'Bal Acc':8s} | {'Recovered':9s} | {'Damaged':8s} | {'Net Diff':8s} | {'Ratio':6s} | {'Gemini':6s} | {'GPT':5s} | {'GLM':5s}")
print("-" * 95)

results = []
for s_w in [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00]:
    for hung in [True, False]:
        r = evaluate_sinkhorn_variation(s_w, target_mix=0.35, use_hungarian=hung)
        r['sinkhorn_w'] = s_w
        r['hungarian'] = hung
        results.append(r)
        h_str = "Active" if hung else "OFF"
        print(f"w = {s_w:4.2f}    | {h_str:9s} | {r['score']:7.4f}  | {r['recovered']:4d} / 1538 | {r['damaged']:4d} / 1161| {r['net']:+5d}    | {r['ratio']:5.2f} | {r['gemini_rec']:3d} / 112| {r['gpt_rec']:2d} / 100| {r['glm_rec']:2d} / 47")

df_res = pd.DataFrame(results)
best_net = df_res.sort_values(by='net', ascending=False).iloc[0]
print("\n=== OPTIMAL UNANCHORED CONFIGURATION ===")
print(f"Optimal Sinkhorn Weight: {best_net['sinkhorn_w']:.2f}")
print(f"Hungarian Status       : {'Active' if best_net['hungarian'] else 'OFF'}")
print(f"Balanced Accuracy      : {best_net['score']:.4f} (Base V68: {base_score:.4f}, Delta: {best_net['score']-base_score:+.4f})")
print(f"Core Errors Recovered  : {int(best_net['recovered'])} / 1538")
print(f"Collateral Damage      : {int(best_net['damaged'])} / 1161")
print(f"Net Gain (Fixed-Broken): {int(best_net['net']):+d}")
print(f"Safety Ratio           : {best_net['ratio']:.2f} : 1")
print(f"Gemini Recovered       : {int(best_net['gemini_rec'])} / 112")
print(f"GPT Recovered          : {int(best_net['gpt_rec'])} / 100")
print(f"GLM Recovered          : {int(best_net['glm_rec'])} / 47")

