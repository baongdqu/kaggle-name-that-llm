import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax

# Load Data & OOF Base
train = pd.read_csv('data/train.csv')
labels = sorted(train.model.unique())
l2i = {l: i for i, l in enumerate(labels)}
y_true = np.array([l2i[m] for m in train.model.values])
prompts = train.prompt_id.values
p2i = {p: i for i, p in enumerate(np.unique(prompts))}
g_true = np.array([p2i[p] for p in prompts])
K = 18

cache = np.load('lab/cache/base_oof.npz')
base_probs = cache['probs']
evidence = cache['evidence']

# Features
wlens = np.array([len(str(t).split()) for t in train.text.values], dtype=np.float32)
char_lens = np.array([len(str(t)) for t in train.text.values], dtype=np.float32)
avg_wlens = char_lens / np.maximum(wlens, 1.0)

# 1. Pipeline Simulator Function
def run_pipeline(name, w_len=0.20, w_wlen=0.0, min_wlen_size=6, thr_cluster=0.15, llama_boost=False, hungarian_anchor=True):
    p = base_probs.copy()
    if llama_boost:
        # Simulate Llama amplified shift (more weight to Llama OOF representation)
        p = np.clip(p ** 1.15, 1e-12, None)
        p /= p.sum(axis=1, keepdims=True)
    
    # Clustering simulation
    if thr_cluster == 0.08:
        # Simulate aggressive false-merges on ~15% adjacent prompt pairs
        g_sim = g_true.copy()
        for i in range(0, len(np.unique(g_sim))-1, 6):
            g_sim[g_sim == i+1] = i
    else:
        g_sim = g_true.copy()
        
    # Relative Length calibration
    LEN_BIAS = np.array([-0.0306, -0.0011, +0.1664, +0.1764, +0.0556, +0.0807,
                         -0.0362, +0.0387, -0.2429, -0.0667, +0.0950, -0.0619,
                         +0.0085, +0.0521, -0.0236, +0.0372, -0.1470, -0.1007], dtype=np.float32)
    WLEN_BIAS = np.array([-0.1886, +0.3526, -0.0640, -1.1015, +0.5982, -0.0520,
                          +0.5204, +1.1509, +1.0350, -0.2390, -0.1549, -0.7586,
                          +0.1915, -0.0895, -0.1149, -0.6532, -0.3990, -0.0348], dtype=np.float32)
    
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
                
    # Soft dispersion
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
                    
    # Damped Sinkhorn 0.45
    q = p_mod.copy()
    target = np.bincount(y_true, minlength=18).astype(np.float64) / len(y_true)
    for _ in range(60):
        q *= (len(q) * target[None, :]) / np.clip(q.sum(0, keepdims=True), 1e-12, None)
        q /= np.clip(q.sum(1, keepdims=True), 1e-12, None)
    tot = 0.55 * np.log(p_mod + 1e-12) + 0.45 * np.log(q + 1e-12)
    
    # Hungarian
    pred = tot.argmax(axis=1).copy()
    if hungarian_anchor:
        c_sizes = np.bincount(g_sim)[g_sim]
        dense = np.unique(g_sim[(c_sizes >= 5) & (c_sizes <= 18)])
        probs_tot = softmax(tot, axis=1)
        for g in dense:
            i_g = np.where(g_sim == g)[0]
            if len(i_g) <= 1: continue
            sub_p = probs_tot[i_g]
            sub_log = tot[i_g]
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
                avail = [c for c in range(K) if c not in assigned]
                if len(avail) >= len(unassigned):
                    cost = -sub_log[unassigned][:, avail]
                    r_ind, c_ind = linear_sum_assignment(cost)
                    for r, c in zip(r_ind, c_ind):
                        pred[i_g[unassigned[r]]] = avail[c]
    return pred

# Execute Pipelines
print("Running simulations on full 2,699 samples...")
preds = {}
preds['V50_Raw'] = run_pipeline('V50', w_len=0.0, w_wlen=0.0, hungarian_anchor=False)
preds['V59_Relative'] = run_pipeline('V59', w_len=0.20, w_wlen=0.0, hungarian_anchor=False)
preds['V62_V63_Anchor'] = run_pipeline('V63', w_len=0.20, w_wlen=0.0, hungarian_anchor=True)
preds['V65_LlamaBoost'] = run_pipeline('V65', w_len=0.20, w_wlen=0.0, llama_boost=True, hungarian_anchor=True)
preds['V66_V67_FalseMerge'] = run_pipeline('V66', w_len=0.20, w_wlen=0.0, thr_cluster=0.08, hungarian_anchor=True)
preds['V68_Orthogonal'] = run_pipeline('V68', w_len=0.20, w_wlen=0.05, min_wlen_size=6, hungarian_anchor=True)

# Compute Accuracy
print("\n=== BALANCED ACCURACY (MACRO RECALL) ===")
for k, pr in preds.items():
    sc = balanced_accuracy_score(y_true, pr)
    print(f"  {k:22s}: {sc:.4f} (Total errors: {(pr != y_true).sum()})")

# Error Masks
err_masks = {k: (pr != y_true) for k, pr in preds.items()}

# Core Stubborn Errors (Intersection across all top models)
top_keys = ['V59_Relative', 'V62_V63_Anchor', 'V68_Orthogonal']
stubborn_mask = np.ones(len(train), dtype=bool)
for k in top_keys:
    stubborn_mask &= err_masks[k]

n_stubborn = stubborn_mask.sum()
avg_errors = np.mean([err_masks[k].sum() for k in top_keys])
overlap_pct = (n_stubborn / avg_errors) * 100

print(f"\n=== ERROR OVERLAP DIAGNOSTIC ===")
print(f"Total dataset size: {len(train)}")
print(f"Average errors across V59/V63/V68: {avg_errors:.1f}")
print(f"Core Stubborn Errors (Failed in ALL 3 versions): {n_stubborn} samples ({overlap_pct:.2f}% overlap!)")

# Analyze Stubborn Errors
stubborn_df = train[stubborn_mask].copy()
stubborn_df['wlen'] = wlens[stubborn_mask]
stubborn_df['avg_wlen'] = avg_wlens[stubborn_mask]

# Intra-family vs Inter-family breakdown
FAMILIES = {
    'claude': [0, 1, 2, 3, 4],
    'gemini': [5, 6, 7, 8],
    'glm': [9, 10],
    'gpt': [11, 12, 13, 14, 15],
    'kimi': [16],
    'qwen': [17]
}
LABEL_TO_FAMILY = {}
for fam, indices in FAMILIES.items():
    for idx in indices:
        LABEL_TO_FAMILY[idx] = fam
intra_cnt = 0
inter_cnt = 0
confusion_pairs = {}

pred_v68 = preds['V68_Orthogonal']
for idx in np.where(stubborn_mask)[0]:
    t_cls = y_true[idx]
    p_cls = pred_v68[idx]
    pair = f"{labels[t_cls]} -> {labels[p_cls]}"
    confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    if LABEL_TO_FAMILY.get(t_cls) == LABEL_TO_FAMILY.get(p_cls):
        intra_cnt += 1
    else:
        inter_cnt += 1

print(f"\n=== STUBBORN ERROR BREAKDOWN ({n_stubborn} samples) ===")
print(f"  Intra-family (Sibling Confusion): {intra_cnt} ({intra_cnt/n_stubborn*100:.1f}%)")
print(f"  Inter-family (Cross-Family Confusion): {inter_cnt} ({inter_cnt/n_stubborn*100:.1f}%)")

print("\n=== TOP 10 WORST CONFUSION PAIRS IN STUBBORN SET ===")
sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: -x[1])[:10]
for p, c in sorted_pairs:
    print(f"  {p:45s}: {c:3d} errors")

print("\n=== LENGTH DISTRIBUTION OF STUBBORN SAMPLES ===")
print(f"  Short texts (< 60 words)   : {(stubborn_df.wlen < 60).sum()} ({((stubborn_df.wlen < 60).sum()/n_stubborn)*100:.1f}%)")
print(f"  Medium texts (60-200 words): {((stubborn_df.wlen >= 60) & (stubborn_df.wlen <= 200)).sum()} ({(((stubborn_df.wlen >= 60) & (stubborn_df.wlen <= 200)).sum()/n_stubborn)*100:.1f}%)")
print(f"  Long texts (> 200 words)   : {(stubborn_df.wlen > 200).sum()} ({((stubborn_df.wlen > 200).sum()/n_stubborn)*100:.1f}%)")
print(f"  Mean length: {stubborn_df.wlen.mean():.1f} words (vs {wlens.mean():.1f} in full dataset)")

# Check Swing Samples between V63 and V68
v63_err = err_masks['V62_V63_Anchor']
v68_err = err_masks['V68_Orthogonal']
fixed_by_v68 = np.where(v63_err & ~v68_err)[0]
broken_by_v68 = np.where(~v63_err & v68_err)[0]
print(f"\n=== SWING SAMPLES V63 vs V68 ===")
print(f"  Samples Fixed by V68: {len(fixed_by_v68)}")
print(f"  Samples Broken by V68: {len(broken_by_v68)}")
print(f"  Net Delta: {len(fixed_by_v68) - len(broken_by_v68):+d}")
