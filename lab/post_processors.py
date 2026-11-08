"""
NTL Digital Twin Lab: Post-Processors & Algorithms Library
Implements Marginal Sinkhorn, Evidence Clipping, Anchor-Protected Kuhn-Munkres,
Soft Injective Dispersion, and Prompt-Relative Length Bias.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax

LABELS = [
    'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'glm-5.3', 'glm-5.3-flash',
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'kimi-k3', 'qwen3.8-max'
]
L2I = {l: i for i, l in enumerate(LABELS)}

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


def marginal_sinkhorn(P, sinkhorn_reg=0.60, target_mix=0.35, max_iter=80, eps=1e-9):
    """
    Damped Marginal Sinkhorn-Knopp algorithm.
    Balances column sums towards uniform N/K = 1/18 with damping.
    """
    if sinkhorn_reg <= 0:
        return P.copy()
        
    P = np.maximum(P, eps)
    N, K = P.shape
    target_col = np.ones(K, dtype=np.float64) / K
    
    # Apply damping/mix to target
    current_col = P.sum(axis=0) / P.sum()
    damped_target = (1.0 - target_mix) * current_col + target_mix * target_col
    target_col = (1.0 - sinkhorn_reg) * current_col + sinkhorn_reg * damped_target
    target_col = target_col / target_col.sum()
    
    # Alternate row and column normalizations
    u = np.ones(N, dtype=np.float64)
    v = np.ones(K, dtype=np.float64)
    
    for _ in range(max_iter):
        # Row normalization
        u = 1.0 / np.maximum(P.dot(v), eps)
        # Col normalization towards target_col * N
        v = (target_col * N) / np.maximum(P.T.dot(u), eps)
        
    P_balanced = u[:, None] * P * v[None, :]
    P_balanced = P_balanced / np.maximum(P_balanced.sum(axis=1, keepdims=True), eps)
    return P_balanced


def evidence_clipping_gate(raw_logits, evidence, fw_sum=1.0, clip_val=1.2,
                           margin_thr=0.60, word_counts=None, active_markers=None):
    """
    Judith007 & Champion V59/V61 Evidence Clipping & Multi-factor Confidence Gate.
    """
    _evidence = np.clip(clip_val * evidence / max(float(fw_sum), 1e-6), -0.30, 0.30)
    _probs = softmax(raw_logits, axis=1)
    _sorted = np.sort(_probs, axis=1)
    _margin = _sorted[:, -1] - _sorted[:, -2]
    _confidence_gate = np.clip((margin_thr - _margin) / margin_thr, 0.0, 1.0)
    
    if word_counts is not None:
        _length_gate = np.clip(np.sqrt(np.maximum(word_counts, 1.0) / 80.0), 0.40, 1.0)
    else:
        _length_gate = 1.0
        
    if active_markers is not None:
        _coverage_gate = np.clip(np.sqrt((active_markers + 1.0) / 4.0), 0.45, 1.0)
    else:
        _coverage_gate = 1.0
        
    _gate = _confidence_gate * _length_gate * _coverage_gate
    _safe_logits = raw_logits + _gate[:, None] * _evidence
    return _safe_logits


def anchor_hungarian_assignment(probs, groups, anchor_margin=0.35, anchor_prob=0.50):
    """
    Latin-Square Anchor-Protected Kuhn-Munkres Hungarian Matching.
    Locks high-confidence predictions in each prompt group, then assigns remaining items 1-1.
    """
    N, K = probs.shape
    final_preds = probs.argmax(axis=1)
    
    unique_groups = np.unique(groups)
    score_matrix = np.log(np.maximum(probs, 1e-12))
    
    for g in unique_groups:
        idx_group = np.where(groups == g)[0]
        group_size = len(idx_group)
        if group_size <= 1:
            continue
            
        sub_probs = probs[idx_group]
        sub_scores = score_matrix[idx_group]
        
        # Sort probabilities to find top-2 margins
        sorted_p = np.sort(sub_probs, axis=1)
        margins = sorted_p[:, -1] - sorted_p[:, -2]
        top_p = sorted_p[:, -1]
        top_cls = sub_scores.argmax(axis=1)
        
        # Identify anchors
        is_anchor = (margins >= anchor_margin) & (top_p >= anchor_prob)
        assigned_classes = set()
        anchor_indices = []
        
        # Greedily lock anchors from highest confidence down
        for local_idx in np.argsort(-top_p):
            if is_anchor[local_idx]:
                cls_id = top_cls[local_idx]
                if cls_id not in assigned_classes:
                    assigned_classes.add(cls_id)
                    anchor_indices.append(local_idx)
                    
        # Solve bipartite matching for unassigned items
        unassigned_indices = [i for i in range(group_size) if i not in anchor_indices]
        if unassigned_indices:
            avail_classes = [c for c in range(K) if c not in assigned_classes]
            if len(avail_classes) >= len(unassigned_indices):
                cost = -sub_scores[unassigned_indices][:, avail_classes]
                row_ind, col_ind = linear_sum_assignment(cost)
                for r, c in zip(row_ind, col_ind):
                    chosen_class = avail_classes[c]
                    final_preds[idx_group[unassigned_indices[r]]] = chosen_class
                    
    return final_preds


def soft_injective_dispersion(probs, groups, damp=0.65):
    """
    V59 Soft Injective Collision Dispersion.
    Softly dampens duplicate predictions inside prompt clusters without hard bipartite forcing.
    """
    P_out = probs.copy()
    unique_groups = np.unique(groups)
    
    for g in unique_groups:
        idx_group = np.where(groups == g)[0]
        if len(idx_group) <= 1:
            continue
            
        preds = P_out[idx_group].argmax(axis=1)
        # Check for duplicate predicted classes
        unique_c, counts = np.unique(preds, return_counts=True)
        for c, count in zip(unique_c, counts):
            if count > 1:
                # Find which row has highest probability
                collided = idx_group[preds == c]
                best_row = collided[np.argmax(P_out[collided, c])]
                # Dampen the others
                for other_row in collided:
                    if other_row != best_row:
                        P_out[other_row, c] *= (1.0 - damp)
                        # Renormalize
                        P_out[other_row] /= P_out[other_row].sum()
                        
    return P_out


def prompt_relative_length_bias(probs, word_counts, groups, beta_per_class=None):
    """
    V59 Prompt-Aware Relative Length Calibration.
    Applies class-conditional empirical length bias based on intra-cluster z-score.
    """
    if beta_per_class is None:
        # Default empirical length coefficients (positive = model writes longer than prompt mean)
        beta_per_class = np.array([
            0.15, -0.10, -0.05, 0.20, 0.10,   # claude: fable, opus46, opus48, opus5, sonnet5
            -0.15, -0.25, -0.20, -0.18,        # gemini: pro, 36flash, 37flash, 38flash
            0.12, 0.08,                        # glm: 5.3, 5.3flash
            0.05, -0.05, 0.02, 0.08, 0.25,     # gpt: 5.5, luna, sol, terra, astra
            -0.10,                             # kimi-k3
            0.18                               # qwen3.8-max
        ], dtype=np.float64)
        
    logits = np.log(np.maximum(probs, 1e-12))
    unique_groups = np.unique(groups)
    
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        if len(idx) <= 1:
            continue
        lengths = word_counts[idx]
        mean_len = np.mean(lengths)
        std_len = np.std(lengths) + 1.0
        delta_z = (lengths - mean_len) / std_len
        logits[idx] += delta_z[:, None] * beta_per_class[None, :]
        
    return softmax(logits, axis=1)

