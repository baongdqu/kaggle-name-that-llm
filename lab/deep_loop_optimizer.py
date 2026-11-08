"""
NTL Digital Twin Lab: Deep Iteration 2 (Triết Lý Vòng Lặp Sâu)
Explores Fine-grained Anchor Kuhn-Munkres, Cluster-Size Gating, and LightGBM Meta-Stacking.
"""
import sys, os, time
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.optimize import linear_sum_assignment
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab.engine import SimulationEngine
from lab.metrics import balanced_accuracy_score, bootstrap_confidence_interval, compute_confusion_analysis
from lab.post_processors import marginal_sinkhorn, soft_injective_dispersion
from lab.loop_optimizer import extract_hierarchical_typography


def run_deep_loop():
    print("=" * 80)
    print("      NTL DIGITAL TWIN SIMULATION LAB: DEEP ITERATIVE OPTIMIZATION (LOOP 2)")
    print("=" * 80)
    
    t_start = time.time()
    engine = SimulationEngine()
    engine.build_or_load_base_oof()
    
    y_true = engine.y_true
    groups = engine.groups
    texts = engine.texts
    word_counts = engine.word_counts
    base_probs = engine.base_probs
    base_logits = np.log(np.maximum(base_probs, 1e-12))
    
    # 1. Apply Best Typography
    typo_evidence, fw, active_markers = extract_hierarchical_typography(texts, y_true)
    fw_sum = max(float(fw.sum()), 1e-6)
    clipped_ev = np.clip(2.0 * typo_evidence / fw_sum, -0.30, 0.30)
    sorted_p = np.sort(base_probs, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    conf_gate = np.clip((0.40 - margin) / 0.40, 0.0, 1.0)
    len_gate = np.clip(np.sqrt(np.maximum(word_counts, 1.0) / 80.0), 0.40, 1.0)
    cov_gate = np.clip(np.sqrt((active_markers + 1.0) / 4.0), 0.45, 1.0)
    gate = conf_gate * len_gate * cov_gate
    safe_logits = base_logits + gate[:, None] * clipped_ev
    
    # 2. Apply Length Z-Score Bias
    base_beta = np.array([
        0.15, -0.10, -0.05, 0.20, 0.10, -0.15, -0.25, -0.20, -0.18,
        0.12, 0.08, 0.05, -0.05, 0.02, 0.08, 0.25, -0.10, 0.18
    ], dtype=np.float64)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) > 1:
            lengths = word_counts[idx]
            delta_z = (lengths - np.mean(lengths)) / (np.std(lengths) + 1.0)
            safe_logits[idx] += delta_z[:, None] * base_beta[None, :]
    probs = softmax(safe_logits, axis=1)
    
    # 3. Apply Soft Dispersion (damp = 0.25)
    probs = soft_injective_dispersion(probs, groups, damp=0.25)
    
    # 4. Apply Marginal Sinkhorn (reg = 0.15, mix = 0.45)
    probs = marginal_sinkhorn(probs, sinkhorn_reg=0.15, target_mix=0.45)
    
    pre_hung_score = balanced_accuracy_score(y_true, probs.argmax(axis=1))
    print(f"[Loop 2 Initial State] Pre-Hungarian Optimized Score: {pre_hung_score:.4f}")
    
    # 5. Deep Scan on Anchor Kuhn-Munkres with Cluster-Size Gating
    print("\n[Loop 2 Sweep] Scanning Fine-Grained Anchor Margins, Probabilities & Cluster Sizes...")
    best_deep_score = pre_hung_score
    best_deep_params = {}
    best_preds = probs.argmax(axis=1)
    
    N, K = probs.shape
    scores = np.log(np.maximum(probs, 1e-12))
    
    # Fine grid
    margins = [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]
    probs_thresh = [0.25, 0.30, 0.35, 0.38, 0.40, 0.45]
    min_cluster_sizes = [1, 6, 10, 12, 14, 16]
    
    count = 0
    for min_size in min_cluster_sizes:
        for m_th in margins:
            for p_th in probs_thresh:
                count += 1
                y_pred = probs.argmax(axis=1).copy()
                
                for g in np.unique(groups):
                    idx_g = np.where(groups == g)[0]
                    g_size = len(idx_g)
                    if g_size < min_size:
                        continue
                        
                    sub_p = probs[idx_g]
                    sub_s = scores[idx_g]
                    
                    sorted_p = np.sort(sub_p, axis=1)
                    mg = sorted_p[:, -1] - sorted_p[:, -2]
                    tp = sorted_p[:, -1]
                    tc = sub_s.argmax(axis=1)
                    
                    is_anchor = (mg >= m_th) & (tp >= p_th)
                    assigned = set()
                    anchors = []
                    
                    for i_loc in np.argsort(-tp):
                        if is_anchor[i_loc]:
                            c = tc[i_loc]
                            if c not in assigned:
                                assigned.add(c)
                                anchors.append(i_loc)
                                
                    unassigned = [i for i in range(g_size) if i not in anchors]
                    if unassigned:
                        avail = [c for c in range(K) if c not in assigned]
                        if len(avail) >= len(unassigned):
                            cost = -sub_s[unassigned][:, avail]
                            r_ind, c_ind = linear_sum_assignment(cost)
                            for r, c in zip(r_ind, c_ind):
                                y_pred[idx_g[unassigned[r]]] = avail[c]
                                
                cur_score = balanced_accuracy_score(y_true, y_pred)
                if cur_score > best_deep_score:
                    best_deep_score = cur_score
                    best_deep_params = {
                        'min_size': min_size,
                        'margin': m_th,
                        'prob': p_th
                    }
                    best_preds = y_pred
                    
    print(f"  -> Scanned {count} parameter triples in {time.time()-t_start:.2f}s!")
    print(f"  -> NEW GLOBAL RECORD: {best_deep_score:.4f} (+{best_deep_score - pre_hung_score:+.4f})")
    print(f"  -> Optimal Configuration: Min Cluster Size = {best_deep_params['min_size']}, Margin = {best_deep_params['margin']:.2f}, Prob = {best_deep_params['prob']:.2f}")
    
    # 6. Monte Carlo Bootstrap on Peak Result
    print("\n[Loop 2 Verification] Running 1,000 Bootstrap Trials on Peak Record...")
    ci = bootstrap_confidence_interval(y_true, best_preds, n_bootstraps=1000)
    print(f"  -> Peak Score: {best_deep_score:.4f}")
    print(f"  -> 95% Confidence Interval: [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] (+/- {ci['noise_floor_pm']:.4f})")
    
    diag = compute_confusion_analysis(y_true, best_preds)
    print("\n[Final Peak Diagnostics]:")
    print(f"  * Inter-Family Accuracy (Hãng): {diag['inter_family_accuracy']*100:.2f}%")
    print("  * Intra-Family Accuracies (Mô hình con):")
    for fam, acc in diag['intra_family_accuracy'].items():
        print(f"    - {fam.capitalize():8s}: {acc*100:.2f}%")
        
    print("\n" + "=" * 80)
    print(f"  DEEP OPTIMIZATION LOOP 2 FINISHED IN {time.time()-t_start:.2f}s!")
    print("=" * 80)
    return best_deep_params, best_deep_score


if __name__ == '__main__':
    run_deep_loop()

