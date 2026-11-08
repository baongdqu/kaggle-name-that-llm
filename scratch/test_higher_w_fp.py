from test_fingerprint_integration import tot_base, oof_fingerprint_logodds, solve_hungarian, base_preds, base_acc, y_true, balanced_accuracy_score
import numpy as np

print("=== SWEEPING HIGHER WEIGHTS w_fp IN [0.10, 0.30] ===")
for w in [0.09, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
    tot_mod = tot_base.copy()
    from test_fingerprint_integration import SIBLINGS
    for name, group in SIBLINGS.items():
        top_is_group = np.isin(tot_base.argmax(axis=1), group)
        for c in group:
            tot_mod[top_is_group, c] += w * oof_fingerprint_logodds[top_is_group, c]
            
    pr = solve_hungarian(tot_mod)
    acc = balanced_accuracy_score(y_true, pr)
    n_diff = (pr != base_preds).sum()
    n_fixed = ((pr == y_true) & (base_preds != y_true)).sum()
    n_broken = ((pr != y_true) & (base_preds == y_true)).sum()
    net = n_fixed - n_broken
    print(f"w_fp={w:.3f}: Acc={acc:.4f} (Delta={acc-base_acc:+.4f}) | Diff={n_diff:3d} | Fixed={n_fixed:3d} | Broken={n_broken:3d} | Net={net:+3d}")

