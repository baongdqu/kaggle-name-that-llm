import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from scipy.optimize import linear_sum_assignment

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

# Feature calculations
sub = train.copy()
sub['wlen'] = sub.text.apply(lambda t: len(str(t).split()))
sub['char_len'] = sub.text.apply(lambda t: len(str(t)))
sub['avg_wlen'] = sub.char_len / np.maximum(sub['wlen'], 1)
sub['ttr'] = sub.text.apply(lambda t: len(set(str(t).lower().split())) / max(1, len(str(t).split())))

# Precompute biases
z_len_full = np.zeros(len(train), dtype=np.float32)
z_wlen_full = np.zeros(len(train), dtype=np.float32)
z_ttr_full = np.zeros(len(train), dtype=np.float32)

for p in np.unique(prompts):
    idx = np.where(prompts == p)[0]
    if len(idx) > 1:
        l_ = sub.loc[idx, 'wlen'].values; s_l = l_.std()
        if s_l > 1e-3: z_len_full[idx] = (l_ - l_.mean()) / s_l
        
        w_ = sub.loc[idx, 'avg_wlen'].values; s_w = w_.std()
        if s_w > 1e-3: z_wlen_full[idx] = (w_ - w_.mean()) / s_w
        
        t_ = sub.loc[idx, 'ttr'].values; s_t = t_.std()
        if s_t > 1e-3: z_ttr_full[idx] = (t_ - t_.mean()) / s_t

b_len = np.array([z_len_full[y_true == i].mean() for i in range(K)], dtype=np.float32)
b_wlen = np.array([z_wlen_full[y_true == i].mean() for i in range(K)], dtype=np.float32)
b_ttr = np.array([z_ttr_full[y_true == i].mean() for i in range(K)], dtype=np.float32)

def evaluate_v68(w_len=0.08, w_wlen=0.06, w_ttr=0.03, n_scenarios=50):
    np.random.seed(42)
    deltas = []
    scores_v63 = []
    scores_v68 = []
    
    for s in range(n_scenarios):
        # 40% slice
        keep = np.random.rand(len(train)) < 0.40
        idx_test = np.where(keep)[0]
        y_test = y_true[idx_test]
        g_test = g_true[idx_test]
        p_test = base_probs[idx_test].copy()
        
        # V63 Baseline: relative length only + Latin-square anchor
        l_test = sub.loc[idx_test, 'wlen'].values
        z_l = np.zeros(len(idx_test), dtype=np.float32)
        z_w = np.zeros(len(idx_test), dtype=np.float32)
        z_t = np.zeros(len(idx_test), dtype=np.float32)
        w_test = sub.loc[idx_test, 'avg_wlen'].values
        t_test = sub.loc[idx_test, 'ttr'].values
        
        for g in np.unique(g_test):
            i_g = np.where(g_test == g)[0]
            if len(i_g) >= 3:
                s_l = l_test[i_g].std()
                if s_l > 1.0: z_l[i_g] = (l_test[i_g] - l_test[i_g].mean()) / s_l
                s_w = w_test[i_g].std()
                if s_w > 0.1: z_w[i_g] = (w_test[i_g] - w_test[i_g].mean()) / s_w
                s_t = t_test[i_g].std()
                if s_t > 0.01: z_t[i_g] = (t_test[i_g] - t_test[i_g].mean()) / s_t
                
        # V63 logits
        log_v63 = np.log(np.maximum(p_test, 1e-12)) + (z_l[:, None] * b_len[None, :]) * w_len
        p_v63 = np.exp(log_v63 - log_v63.max(axis=1, keepdims=True))
        p_v63 /= p_v63.sum(axis=1, keepdims=True)
        
        # V68 logits: adds relative word length + relative TTR
        log_v68 = log_v63 + (z_w[:, None] * b_wlen[None, :]) * w_wlen + (z_t[:, None] * b_ttr[None, :]) * w_ttr
        p_v68 = np.exp(log_v68 - log_v68.max(axis=1, keepdims=True))
        p_v68 /= p_v68.sum(axis=1, keepdims=True)
        
        # Run Anchor Hungarian >= 5 on both
        def run_anchor(p_mat, log_mat):
            pred = p_mat.argmax(axis=1).copy()
            c_sizes = np.bincount(g_test)[g_test]
            dense = np.unique(g_test[(c_sizes >= 5) & (c_sizes <= 18)])
            for g in dense:
                i_g = np.where(g_test == g)[0]
                if len(i_g) <= 1: continue
                sub_p = p_mat[i_g]; sub_log = log_mat[i_g]
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
            return balanced_accuracy_score(y_test, pred)
            
        sc_63 = run_anchor(p_v63, log_v63)
        sc_68 = run_anchor(p_v68, log_v68)
        scores_v63.append(sc_63)
        scores_v68.append(sc_68)
        deltas.append(sc_68 - sc_63)
        
    return np.mean(scores_v63), np.mean(scores_v68), np.mean(deltas), (np.array(deltas) > 0).mean() * 100

print("Simulating V68 Tri-Attribute Prompt Calibration vs V63 on 50 Scenarios...", flush=True)
sc63, sc68, gain, win = evaluate_v68(w_len=0.08, w_wlen=0.06, w_ttr=0.03, n_scenarios=50)
print(f"Mean V63 Base Score : {sc63:.4f}", flush=True)
print(f"Mean V68 Tri-Score  : {sc68:.4f}", flush=True)
print(f"Gain over V63       : {gain:+.4f} ({gain*100:+.2f}%)", flush=True)
print(f"Head-to-head Win Rate: {win:.1f}%", flush=True)

