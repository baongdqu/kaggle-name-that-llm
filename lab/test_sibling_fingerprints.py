import sys, os
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score

from test_llm_fingerprints import F_raw, F_rel, y_true, prompts, labels, l2i

print("=== INTRA-FAMILY SIBLING DISCRIMINATION WITH RELATIVE FINGERPRINTS ===")

gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(F_raw, y_true, groups=prompts))

SIBLINGS = {
    'gemini_flash': [l2i['gemini-3.6-flash'], l2i['gemini-3.7-flash'], l2i['gemini-3.8-flash']],
    'glm53': [l2i['glm-5.3'], l2i['glm-5.3-flash']],
    'gpt56': [l2i['gpt-5.6-luna'], l2i['gpt-5.6-sol'], l2i['gpt-5.6-terra']],
    'claude_opus': [l2i['claude-opus-4-6'], l2i['claude-opus-4-8'], l2i['claude-opus-5']]
}

for name, group in SIBLINGS.items():
    mask = np.isin(y_true, group)
    y_g = np.array([group.index(y) if y in group else -1 for y in y_true])
    
    # 1. Using Raw Fingerprints
    accs_raw = []
    # 2. Using Per-Prompt Relative Features
    accs_rel = []
    # 3. Combined Raw + Rel
    accs_comb = []
    
    F_comb = np.hstack([F_raw, F_rel])
    
    for tr_idx, va_idx in folds:
        tr_m = tr_idx[np.isin(y_true[tr_idx], group)]
        va_m = va_idx[np.isin(y_true[va_idx], group)]
        if len(va_m) == 0: continue
        
        clf1 = LogisticRegression(C=1.0, max_iter=300, random_state=42)
        clf1.fit(F_raw[tr_m], y_g[tr_m])
        accs_raw.append((clf1.predict(F_raw[va_m]) == y_g[va_m]).mean())
        
        clf2 = LogisticRegression(C=1.0, max_iter=300, random_state=42)
        clf2.fit(F_rel[tr_m], y_g[tr_m])
        accs_rel.append((clf2.predict(F_rel[va_m]) == y_g[va_m]).mean())
        
        clf3 = LogisticRegression(C=1.0, max_iter=300, random_state=42)
        clf3.fit(F_comb[tr_m], y_g[tr_m])
        accs_comb.append((clf3.predict(F_comb[va_m]) == y_g[va_m]).mean())
        
    random_guess = 1.0 / len(group)
    print(f"\nSibling Group [{name:13s}] (Classes: {len(group)}, Random Baseline: {random_guess:.4f}):")
    print(f"  Raw Fingerprints Acc      : {np.mean(accs_raw):.4f} (Delta: {np.mean(accs_raw)-random_guess:+.4f})")
    print(f"  Intra-Prompt Relative Acc : {np.mean(accs_rel):.4f} (Delta: {np.mean(accs_rel)-random_guess:+.4f})")
    print(f"  Combined (Raw + Rel) Acc  : {np.mean(accs_comb):.4f} (Delta: {np.mean(accs_comb)-random_guess:+.4f})")

