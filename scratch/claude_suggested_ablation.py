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
# Root TTR (Guiraud's R = V / sqrt(N)) to eliminate length dependency!
sub['root_ttr'] = sub.text.apply(lambda t: len(set(str(t).lower().split())) / np.sqrt(max(1, len(str(t).split()))))

# Check correlation across full dataset
print("=== CORRELATION MATRIX (Checking Multicollinearity) ===")
corr = sub[['wlen', 'avg_wlen', 'root_ttr']].corr()
print(corr.to_string())

# Compute relative z-scores per prompt cluster
z_l = np.zeros(len(train), dtype=np.float32)
z_w = np.zeros(len(train), dtype=np.float32)
z_rt = np.zeros(len(train), dtype=np.float32)

for p in np.unique(prompts):
    idx = np.where(prompts == p)[0]
    if len(idx) >= 6:
        l_ = sub.loc[idx, 'wlen'].values; s_l = l_.std()
        if s_l > 1e-3: z_l[idx] = (l_ - l_.mean()) / s_l
        
        w_ = sub.loc[idx, 'avg_wlen'].values; s_w = w_.std()
        if s_w > 1e-3: z_w[idx] = (w_ - w_.mean()) / s_w
        
        t_ = sub.loc[idx, 'root_ttr'].values; s_t = t_.std()
        if s_t > 1e-3: z_rt[idx] = (t_ - t_.mean()) / s_t

# Biases
b_l = np.array([z_l[y_true == i].mean() for i in range(K)], dtype=np.float32)
b_w = np.array([z_w[y_true == i].mean() for i in range(K)], dtype=np.float32)
b_rt = np.array([z_rt[y_true == i].mean() for i in range(K)], dtype=np.float32)

def evaluate_ablation(w_len, w_wlen, w_rttr, min_cluster_size=6, n_scenarios=50):
    np.random.seed(42)
    deltas = []
    
    for s in range(n_scenarios):
        keep = np.random.rand(len(train)) < 0.40
        idx_test = np.where(keep)[0]
        y_test = y_true[idx_test]
        g_test = g_true[idx_test]
        p_test = base_probs[idx_test].copy()
        
        # Test features
        l_test = sub.loc[idx_test, 'wlen'].values
        w_test = sub.loc[idx_test, 'avg_wlen'].values
        t_test = sub.loc[idx_test, 'root_ttr'].values
        
        zl = np.zeros(len(idx_test), dtype=np.float32)
        zw = np.zeros(len(idx_test), dtype=np.float32)
        zrt = np.zeros(len(idx_test), dtype=np.float32)
        
        for g in np.unique(g_test):
            i_g = np.where(g_test == g)[0]
            if len(i_g) >= 3:
                s_l = l_test[i_g].std()
                if s_l > 1.0: zl[i_g] = (l_test[i_g] - l_test[i_g].mean()) / s_l
            if len(i_g) >= min_cluster_size: # Claude safeguard: only for robust clusters
                s_w = w_test[i_g].std()
                if s_w > 0.1: zw[i_g] = (w_test[i_g] - w_test[i_g].mean()) / s_w
                s_t = t_test[i_g].std()
                if s_t > 0.05: zrt[i_g] = (t_test[i_g] - t_test[i_g].mean()) / s_t
                
        # V63 baseline: length only (0.08)
        log_base = np.log(np.maximum(p_test, 1e-12)) + (zl[:, None] * b_l[None, :]) * 0.08
        p_base = np.exp(log_base - log_base.max(axis=1, keepdims=True)); p_base /= p_base.sum(axis=1, keepdims=True)
        
        # Candidate
        log_cand = np.log(np.maximum(p_test, 1e-12)) + (zl[:, None] * b_l[None, :]) * w_len + (zw[:, None] * b_w[None, :]) * w_wlen + (zrt[:, None] * b_rt[None, :]) * w_rttr
        p_cand = np.exp(log_cand - log_cand.max(axis=1, keepdims=True)); p_cand /= p_cand.sum(axis=1, keepdims=True)
        
        # Hungarian evaluator
        def score_mat(p_m, log_m):
            pred = p_m.argmax(axis=1).copy()
            c_sizes = np.bincount(g_test)[g_test]
            dense = np.unique(g_test[(c_sizes >= 5) & (c_sizes <= 18)])
            for g in dense:
                i_g = np.where(g_test == g)[0]
                if len(i_g) <= 1: continue
                sub_p = p_m[i_g]; sub_log = log_m[i_g]
                sort_p = np.sort(sub_p, axis=1)
                margins = sort_p[:, -1] - sort_p[:, -2]
                top_p = sort_p[:, -1]; top_c = sub_log.argmax(axis=1)
                is_anc = (margins >= 0.20) & (top_p >= 0.35)
                assigned = set(); anc_idx = []
                for l_idx in np.argsort(-top_p):
                    if is_anc[l_idx]:
                        cls_ = top_c[l_idx]
                        if cls_ not in assigned: assigned.add(cls_); anc_idx.append(l_idx)
                unassigned = [i for i in range(len(i_g)) if i not in anc_idx]
                if unassigned:
                    avail = [c for c in range(K) if c not in assigned]
                    if len(avail) >= len(unassigned):
                        cost = -sub_log[unassigned][:, avail]
                        r_ind, c_ind = linear_sum_assignment(cost)
                        for r, c in zip(r_ind, c_ind): pred[i_g[unassigned[r]]] = avail[c]
            return balanced_accuracy_score(y_test, pred)
            
        sc_b = score_mat(p_base, log_base)
        sc_c = score_mat(p_cand, log_cand)
        deltas.append(sc_c - sc_b)
        
    return np.mean(deltas), (np.array(deltas) > 0).mean() * 100

print("\n=== CLAUDE-RECOMMENDED ABLATION TESTS (50 SCENARIOS) ===")
print("1. V63 Baseline (Length only w=0.08) -> Gain: +0.0000 | Win: 0.0%")

# Ablation A: Average Word Length Alone
gain_w, win_w = evaluate_ablation(w_len=0.08, w_wlen=0.05, w_rttr=0.00, min_cluster_size=6)
print(f"2. + Avg Word Length Alone (w=0.05, size>=6) -> Gain: {gain_w:+.4f} | Win: {win_w:.1f}%")

# Ablation B: Root TTR Alone
gain_t, win_t = evaluate_ablation(w_len=0.08, w_wlen=0.00, w_rttr=0.03, min_cluster_size=6)
print(f"3. + Root-TTR Alone (w=0.03, size>=6)        -> Gain: {gain_t:+.4f} | Win: {win_t:.1f}%")

# Ablation C: Both combined with Claude Safeguard
gain_c, win_c = evaluate_ablation(w_len=0.08, w_wlen=0.05, w_rttr=0.02, min_cluster_size=6)
print(f"4. Combo (Avg Word Length + Root-TTR)        -> Gain: {gain_c:+.4f} | Win: {win_c:.1f}%")

