import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

train = pd.read_csv('data/train.csv')
prompts_raw = train.prompt_id.values
p2i = {p: i for i, p in enumerate(np.unique(prompts_raw))}
true_prompts = np.array([p2i[p] for p in prompts_raw])

models = train.model.values
labels = sorted(train.model.unique())
l2i = {l: i for i, l in enumerate(labels)}
y_true = np.array([l2i[m] for m in models])
K = 18

cache = np.load('lab/cache/base_oof.npz')
base_probs = cache['probs']

def fast_capped_agglo(S, cap=18, thr=0.08):
    n = len(S); A = S.astype(np.float32).copy(); np.fill_diagonal(A, -np.inf)
    size = np.ones(n, np.int32); alive = np.ones(n, bool); lab = np.arange(n)
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        M = np.where(ok, A, -np.inf); k = int(np.argmax(M)); i, j = divmod(k, n)
        if not np.isfinite(M[i, j]) or M[i, j] < thr: break
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new; A[:, i] = new; A[i, i] = -np.inf; A[j] = -np.inf; A[:, j] = -np.inf
        size[i] += size[j]; alive[j] = False; lab[lab == j] = i
    return lab

def run_simulation(v_config, thr, n_scenarios=20):
    np.random.seed(42)
    deltas = []
    aris = []
    
    for s in range(n_scenarios):
        # 40% slice
        keep_mask = np.random.rand(len(train)) < 0.40
        idx_test = np.where(keep_mask)[0]
        y_test = y_true[idx_test]
        p_test = base_probs[idx_test].copy()
        texts_test = train.text.values[idx_test]
        
        # Cluster with vectorizer
        vec = TfidfVectorizer(**v_config)
        Xc = vec.fit_transform(texts_test)
        g_est = fast_capped_agglo((Xc @ Xc.T).toarray(), 18, thr)
        
        ari = adjusted_rand_score(true_prompts[idx_test], g_est)
        aris.append(ari)
        
        pred_free = p_test.argmax(axis=1)
        score_f = balanced_accuracy_score(y_test, pred_free)
        
        # Anchor Hungarian with threshold >= 5
        cluster_sizes = np.bincount(g_est)[g_est]
        dense_clusters = np.unique(g_est[(cluster_sizes >= 5) & (cluster_sizes <= 18)])
        
        pred_anchor = pred_free.copy()
        logits_test = np.log(np.maximum(p_test, 1e-12))
        
        for g in dense_clusters:
            idx_g = np.where(g_est == g)[0]
            gsize = len(idx_g)
            if gsize <= 1:
                continue
            sub_p = p_test[idx_g]
            sub_logits = logits_test[idx_g]
            sorted_p = np.sort(sub_p, axis=1)
            margins = sorted_p[:, -1] - sorted_p[:, -2]
            top_p = sorted_p[:, -1]
            top_cls = sub_logits.argmax(axis=1)
            
            is_anchor = (margins >= 0.20) & (top_p >= 0.35)
            assigned_classes = set()
            anchor_indices = []
            
            for local_idx in np.argsort(-top_p):
                if is_anchor[local_idx]:
                    cls_id = top_cls[local_idx]
                    if cls_id not in assigned_classes:
                        assigned_classes.add(cls_id)
                        anchor_indices.append(local_idx)
                        
            unassigned_indices = [i for i in range(gsize) if i not in anchor_indices]
            if unassigned_indices:
                avail_classes = [c for c in range(K) if c not in assigned_classes]
                if len(avail_classes) >= len(unassigned_indices):
                    cost = -sub_logits[unassigned_indices][:, avail_classes]
                    row_ind, col_ind = linear_sum_assignment(cost)
                    for r, c in zip(row_ind, col_ind):
                        pred_anchor[idx_g[unassigned_indices[r]]] = avail_classes[c]
                        
        score_a = balanced_accuracy_score(y_test, pred_anchor)
        deltas.append(score_a - score_f)
        
    return np.mean(aris), np.mean(deltas), (np.array(deltas) > 0).mean() * 100

print("Testing Baseline V63 Clustering: unigram, max_df=0.05, thr=0.15...")
cfg_v63 = dict(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05)
ari_v63, gain_v63, win_v63 = run_simulation(cfg_v63, 0.15, n_scenarios=20)
print(f"V63 Baseline  -> Mean ARI: {ari_v63:.4f} | Mean Gain: {gain_v63:+.4f} | Win Rate: {win_v63:.1f}%")

print("\nTesting Champion V66 Upgraded Clustering: ngram (1,2), max_df=0.10, thr=0.08...")
cfg_v66 = dict(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.10, ngram_range=(1, 2))
ari_v66, gain_v66, win_v66 = run_simulation(cfg_v66, 0.08, n_scenarios=20)
print(f"V66 Upgraded  -> Mean ARI: {ari_v66:.4f} | Mean Gain: {gain_v66:+.4f} | Win Rate: {win_v66:.1f}%")

