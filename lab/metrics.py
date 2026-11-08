"""
NTL Digital Twin Lab: Metrics & Validation Engine
Calculates exact Balanced Accuracy (Macro-Recall), Intra/Inter-Family breakdown,
and Monte Carlo Bootstrap Confidence Intervals.
"""
import numpy as np

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


def balanced_accuracy_score(y_true, y_pred, n_classes=18):
    """Exact Kaggle competition metric: Macro-Averaged Recall across all classes."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = []
    for c in range(n_classes):
        mask = (y_true == c)
        total = np.sum(mask)
        if total > 0:
            recalls.append(np.sum(y_pred[mask] == c) / float(total))
        else:
            recalls.append(0.0)
    return float(np.mean(recalls))


def compute_confusion_analysis(y_true, y_pred, n_classes=18):
    """
    Detailed diagnostic analysis:
    - Balanced Accuracy (Kaggle score)
    - Inter-Family Accuracy (Vendor separation: Claude vs GPT vs Gemini vs GLM vs Kimi vs Qwen)
    - Intra-Family Accuracy (Sister model distinction: Opus 4.6 vs 4.8, Gemini 3.7 vs 3.8)
    - Per-class recall dictionary
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    overall_acc = balanced_accuracy_score(y_true, y_pred, n_classes=n_classes)
    
    # Inter-family accuracy
    fam_true = np.array([LABEL_TO_FAMILY[y] for y in y_true])
    fam_pred = np.array([LABEL_TO_FAMILY[y] for y in y_pred])
    inter_fam_acc = float(np.mean(fam_true == fam_pred))
    
    # Intra-family accuracies
    intra_fam_accs = {}
    for fam, indices in FAMILIES.items():
        if len(indices) > 1:
            mask = np.isin(y_true, indices) & (fam_true == fam_pred)
            if np.sum(mask) > 0:
                intra_fam_accs[fam] = float(np.mean(y_true[mask] == y_pred[mask]))
            else:
                intra_fam_accs[fam] = 0.0
                
    # Per-class recalls
    per_class_recall = {}
    for c in range(n_classes):
        mask = (y_true == c)
        if np.sum(mask) > 0:
            per_class_recall[LABELS[c]] = float(np.sum(y_pred[mask] == c) / np.sum(mask))
        else:
            per_class_recall[LABELS[c]] = 0.0

    return {
        'balanced_accuracy': overall_acc,
        'inter_family_accuracy': inter_fam_acc,
        'intra_family_accuracy': intra_fam_accs,
        'per_class_recall': per_class_recall
    }


def bootstrap_confidence_interval(y_true, y_pred, n_bootstraps=1000, sample_size=1600, seed=42):
    """
    Monte Carlo Bootstrap Resampling (1,000 resamples of 1,600 test rows).
    Simulates the exact Public Test variance and computes:
    - Expected Mean
    - Standard Deviation (Noise Floor)
    - 95% Confidence Interval [ci_low, ci_high]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    
    scores = np.empty(n_bootstraps, dtype=np.float64)
    for i in range(n_bootstraps):
        idx = rng.choice(n, size=sample_size, replace=True)
        scores[i] = balanced_accuracy_score(y_true[idx], y_pred[idx])
        
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    ci_low = float(np.percentile(scores, 2.5))
    ci_high = float(np.percentile(scores, 97.5))
    
    return {
        'mean': mean,
        'std': std,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'noise_floor_pm': 1.96 * std
    }

