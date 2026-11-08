import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score

train = pd.read_csv('data/train.csv')
prompts_raw = train.prompt_id.values
p2i = {p: i for i, p in enumerate(np.unique(prompts_raw))}
prompts = np.array([p2i[p] for p in prompts_raw])

models = train.model.values
labels = sorted(train.model.unique())
l2i = {l: i for i, l in enumerate(labels)}
y_true = np.array([l2i[m] for m in models])
K = len(labels) # 18

print(f"Total dataset: {len(train)} rows across {len(np.unique(prompts))} prompts")

# Load cached OOF probabilities from lab
cache = np.load('lab/cache/base_oof.npz')
base_probs = cache['probs']

def evaluate_threshold(min_cluster_size=6, anchor_margin=0.20, anchor_prob=0.35, n_scenarios=50):
    np.random.seed(42)
    deltas = []
    scores_free = []
    scores_anchor = []
    active_clusters_count = []
    
    for s in range(n_scenarios):
        # Simulate 40% Public LB slice: randomly keep 40% of rows
        keep_mask = np.random.rand(len(train)) < 0.40
        idx_test = np.where(keep_mask)[0]
        y_test = y_true[idx_test]
        p_test = base_probs[idx_test].copy()
        g_test = prompts[idx_test]
        
        # Free score
        pred_free = p_test.argmax(axis=1)
        score_f = balanced_accuracy_score(y_test, pred_free)
        scores_free.append(score_f)
        
        # Anchor Hungarian with min_cluster_size
        cluster_sizes = np.bincount(g_test)[g_test]
        dense_clusters = np.unique(g_test[(cluster_sizes >= min_cluster_size) & (cluster_sizes <= 18)])
        active_clusters_count.append(len(dense_clusters))
        
        pred_anchor = pred_free.copy()
        logits_test = np.log(np.maximum(p_test, 1e-12))
        
        for g in dense_clusters:
            idx_g = np.where(g_test == g)[0]
            gsize = len(idx_g)
            if gsize <= 1:
                continue
            sub_p = p_test[idx_g]
            sub_logits = logits_test[idx_g]
            sorted_p = np.sort(sub_p, axis=1)
            margins = sorted_p[:, -1] - sorted_p[:, -2]
            top_p = sorted_p[:, -1]
            top_cls = sub_logits.argmax(axis=1)
            
            is_anchor = (margins >= anchor_margin) & (top_p >= anchor_prob)
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
        scores_anchor.append(score_a)
        deltas.append(score_a - score_f)
        
    return {
        'min_size': min_cluster_size,
        'mean_active_clusters': np.mean(active_clusters_count),
        'mean_free': np.mean(scores_free),
        'mean_anchor': np.mean(scores_anchor),
        'mean_gain': np.mean(deltas),
        'win_rate': (np.array(deltas) > 0).mean() * 100,
        'loss_rate': (np.array(deltas) < 0).mean() * 100
    }

print("\n--- Scanning Cluster Thresholds on 40% Public LB Slice (50 Scenarios) ---")
for sz in [12, 10, 8, 6, 5, 4]:
    res = evaluate_threshold(min_cluster_size=sz, n_scenarios=50)
    print(f"Threshold >= {res['min_size']:2d} | Active Clusters: {res['mean_active_clusters']:5.1f} | Mean Score: {res['mean_anchor']:.4f} (Free: {res['mean_free']:.4f}) | Gain: {res['mean_gain']:+.4f} | Win: {res['win_rate']:5.1f}% | Loss: {res['loss_rate']:4.1f}%")
