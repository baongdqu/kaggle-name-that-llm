import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

# Load results from previous run
import sys
sys.path.append('.')
sys.path.append('lab')
from test_mixture_of_specialists import oof_specialist_probs, pred_v68, y_true, SPECIALIST_GROUPS, LABELS

base_score = balanced_accuracy_score(y_true, pred_v68)
print(f"Base V68 Balanced Accuracy: {base_score:.4f}")

# Test selective specialists (Gemini Flash + GLM only)
for conf_thr in [0.45, 0.50, 0.55, 0.60, 0.65]:
    pred_selective = pred_v68.copy()
    n_mod = 0
    for name in ['gemini_flash', 'glm53']: # Exclude gpt56
        group = SPECIALIST_GROUPS[name]
        in_group_mask = np.isin(pred_v68, group)
        target_indices = np.where(in_group_mask)[0]
        for idx in target_indices:
            spec_p = oof_specialist_probs[name][idx]
            if spec_p.sum() > 0:
                chosen_local = spec_p.argmax()
                chosen_global = group[chosen_local]
                if spec_p[chosen_local] >= conf_thr and chosen_global != pred_v68[idx]:
                    pred_selective[idx] = chosen_global
                    n_mod += 1
    score = balanced_accuracy_score(y_true, pred_selective)
    print(f"Selective MoS (Gemini+GLM, conf >= {conf_thr:.2f}): Balanced Acc: {score:.4f} (Delta: {score - base_score:+.4f}, Modified {n_mod} samples)")

