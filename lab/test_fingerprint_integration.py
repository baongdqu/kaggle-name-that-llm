import sys, os
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax

from test_llm_fingerprints import F_raw, F_rel, y_true, prompts, labels, l2i

# Load cache and definition
cache = np.load('lab/cache/base_oof.npz')
base_probs = cache['probs']
p2i = {p: i for i, p in enumerate(np.unique(prompts))}
g_true = np.array([p2i[p] for p in prompts])
wlens = np.array([len(str(t).split()) for t in pd.read_csv('data/train.csv').text.values], dtype=np.float32)
char_lens = np.array([len(str(t)) for t in pd.read_csv('data/train.csv').text.values], dtype=np.float32)
avg_wlens = char_lens / np.maximum(wlens, 1.0)

LEN_BIAS = np.array([-0.0306, -0.0011, +0.1664, +0.1764, +0.0556, +0.0807,
                     -0.0362, +0.0387, -0.2429, -0.0667, +0.0950, -0.0619,
                     +0.0085, +0.0521, -0.0236, +0.0372, -0.1470, -0.1007], dtype=np.float32)
WLEN_BIAS = np.array([-0.1886, +0.3526, -0.0640, -1.1015, +0.5982, -0.0520,
                      +0.5204, +1.1509, +1.0350, -0.2390, -0.1549, -0.7586,
                      +0.1915, -0.0895, -0.1149, -0.6532, -0.3990, -0.0348], dtype=np.float32)

print("=== GLOBAL SIMULATION: V68 ANCHOR + RELATIVE FINGERPRINT HEADS ===")

def get_base_v68_tot():
    p = base_probs.copy()
    g_sim = g_true.copy()
    w_len = 0.20
    w_wlen = 0.05
    min_wlen_size = 6
    
    p_mod = p.copy()
    for g in np.unique(g_sim):
        idx = np.where(g_sim == g)[0]
        gsize = len(idx)
        if gsize >= 3:
            adj = np.zeros((gsize, 18), dtype=np.float32)
            l_sub = wlens[idx]
            s_l = l_sub.std()
            if s_l > 1.0:
                zl = (l_sub - l_sub.mean()) / s_l
                adj += w_len * np.outer(zl, LEN_BIAS)
            if w_wlen > 0 and gsize >= min_wlen_size:
                w_sub = avg_wlens[idx]
                s_w = w_sub.std()
                if s_w > 0.05:
                    zw = (w_sub - w_sub.mean()) / s_w
                    adj += w_wlen * np.outer(zw, WLEN_BIAS)
            if np.any(adj != 0):
                sub_p = p_mod[idx] * np.exp(adj)
                p_mod[idx] = sub_p / (sub_p.sum(axis=1, keepdims=True) + 1e-12)
                
    for g in np.unique(g_sim):
        idx = np.where(g_sim == g)[0]
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
                    
    q = p_mod.copy()
    target = np.bincount(y_true, minlength=18).astype(np.float64) / len(y_true)
    for _ in range(60):
        q *= (len(q) * target[None, :]) / np.clip(q.sum(0, keepdims=True), 1e-12, None)
        q /= np.clip(q.sum(1, keepdims=True), 1e-12, None)
    tot = 0.55 * np.log(p_mod + 1e-12) + 0.45 * np.log(q + 1e-12)
    return tot, g_sim

tot_base, g_sim = get_base_v68_tot()

SIBLINGS = {
    'gemini_flash': [l2i['gemini-3.6-flash'], l2i['gemini-3.7-flash'], l2i['gemini-3.8-flash']],
    'glm53': [l2i['glm-5.3'], l2i['glm-5.3-flash']],
    'gpt56': [l2i['gpt-5.6-luna'], l2i['gpt-5.6-sol'], l2i['gpt-5.6-terra']],
    'claude_opus': [l2i['claude-opus-4-6'], l2i['claude-opus-4-8'], l2i['claude-opus-5']]
}

gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(F_raw, y_true, groups=prompts))
F_comb = np.hstack([F_raw, F_rel])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
F_comb_norm = scaler.fit_transform(F_comb)

oof_fingerprint_logodds = np.zeros((len(y_true), 18), dtype=np.float64)

for name, group in SIBLINGS.items():
    mask = np.isin(y_true, group)
    y_g = np.array([group.index(y) if y in group else -1 for y in y_true])
    for tr_idx, va_idx in folds:
        tr_m = tr_idx[np.isin(y_true[tr_idx], group)]
        va_m = va_idx[np.isin(y_true[va_idx], group)]
        if len(va_m) == 0: continue
        clf = LogisticRegression(C=0.5, max_iter=500, random_state=42)
        clf.fit(F_comb_norm[tr_m], y_g[tr_m])
        probs = clf.predict_proba(F_comb_norm[va_idx])
        log_p = np.log(probs + 1e-9)
        log_p -= log_p.mean(axis=1, keepdims=True)
        for loc_i, glob_c in enumerate(group):
            oof_fingerprint_logodds[va_idx, glob_c] = log_p[:, loc_i]

def solve_hungarian(tot_matrix):
    pred = tot_matrix.argmax(axis=1).copy()
    c_sizes = np.bincount(g_sim)[g_sim]
    dense = np.unique(g_sim[(c_sizes >= 5) & (c_sizes <= 18)])
    probs_tot = softmax(tot_matrix, axis=1)
    
    for g in dense:
        i_g = np.where(g_sim == g)[0]
        if len(i_g) <= 1: continue
        sub_p = probs_tot[i_g]
        sub_log = tot_matrix[i_g]
        sort_p = np.sort(sub_p, axis=1)
        margins = sort_p[:, -1] - sort_p[:, -2]
        top_p = sort_p[:, -1]
        top_c = sub_log.argmax(axis=1)
        
        is_anc = (margins >= 0.20) & (top_p >= 0.35)
        assigned = set(); anc_idx = []
        for l_idx in np.argsort(-top_p):
            if is_anc[l_idx]:
                cls_ = top_c[l_idx]
                if cls_ not in assigned:
                    assigned.add(cls_)
                    anc_idx.append(l_idx)
        unassigned = [i for i in range(len(i_g)) if i not in anc_idx]
        if unassigned:
            avail = [c for c in range(18) if c not in assigned]
            if len(avail) >= len(unassigned):
                cost = -sub_log[unassigned][:, avail]
                r_ind, c_ind = linear_sum_assignment(cost)
                for r, c in zip(r_ind, c_ind):
                    pred[i_g[unassigned[r]]] = avail[c]
    return pred

base_preds = solve_hungarian(tot_base.copy())
base_acc = balanced_accuracy_score(y_true, base_preds)
print(f"Exact Base V68 Balanced Acc: {base_acc:.4f} (Total errors: {(base_preds != y_true).sum()})")

print("\n=== SWEEPING FINGERPRINT INJECTION WEIGHT (w_fp) ===")
for w in [0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 0.080, 0.100]:
    tot_mod = tot_base.copy()
    for name, group in SIBLINGS.items():
        top_is_group = np.isin(tot_base.argmax(axis=1), group)
        for c in group:
            tot_mod[top_is_group, c] += w * oof_fingerprint_logodds[top_is_group, c]
            
    pr = solve_hungarian(tot_mod)
    acc = balanced_accuracy_score(y_true, pr)
    n_diff = (pr != base_preds).sum()
    n_fixed = ((pr == y_true) & (base_preds != y_true)).sum()
    n_broken = ((pr != y_true) & (base_preds == y_true)).sum()
    print(f"w_fp={w:.3f}: Acc={acc:.4f} (Delta={acc-base_acc:+.4f}) | Diff={n_diff:2d} | Fixed={n_fixed:2d} | Broken={n_broken:2d} | Net={n_fixed-n_broken:+2d}")
