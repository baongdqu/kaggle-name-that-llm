import sys
sys.path.append('lab')
from test_v68_exact_spec import get_v68_tot, base_v68, base_acc, y_true, balanced_accuracy_score
import numpy as np

# Let's test fine-grained w_spec in [0.005, 0.010, 0.015, 0.020]
print("=== FINE-GRAINED SWEET SPOT SEARCH FOR HUNGARIAN-AWARE SPECIALIST LOGIT ===")
for w in [0.003, 0.005, 0.008, 0.010, 0.012, 0.015]:
    pr = get_v68_tot(w_spec=w)
    acc = balanced_accuracy_score(y_true, pr)
    n_diff = (pr != base_v68).sum()
    n_fixed = ((pr == y_true) & (base_v68 != y_true)).sum()
    n_broken = ((pr != y_true) & (base_v68 == y_true)).sum()
    print(f"w_spec={w:.3f}: Acc={acc:.4f} (Delta={acc-base_acc:+.4f}) | Diff={n_diff:2d} | Fixed={n_fixed:2d} | Broken={n_broken:2d} | Net={n_fixed-n_broken:+2d}")

