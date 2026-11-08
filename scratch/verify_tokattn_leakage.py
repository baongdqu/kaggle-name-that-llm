import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

train = pd.read_csv('data/train.csv')
labels = sorted(train.model.unique())
y_true = train.model.map({l: i for i, l in enumerate(labels)}).values

tok_deb = np.load('scratch/out_tokattn2/tokattn_debl23_oof.npz')
tok_q8 = np.load('scratch/out_tokattn2/tokattn_q8l22_oof.npz')

p_deb = tok_deb['probs']
p_q8 = tok_q8['probs']
p_tok = 0.5 * p_deb + 0.5 * p_q8

# Check fold splits
print("Fold split check in TokAttn:")
folds = tok_deb['fold']
print("Unique folds:", np.unique(folds, return_counts=True))

# Check prompt grouping per fold
prompt_ids = train.prompt_id.values
fold_per_prompt = {}
leaked = False
for pid, f in zip(prompt_ids, folds):
    if pid not in fold_per_prompt:
        fold_per_prompt[pid] = f
    elif fold_per_prompt[pid] != f:
        leaked = True
        break
print(f"GroupKFold strict check by prompt_id: {'LEAK DETECTED!' if leaked else 'PASSED! 100% strict GroupKFold without leakage.'}")

# Check per-fold balanced accuracy
print("\nPer-fold Balanced Accuracy:")
for k in range(5):
    val_idx = (folds == k)
    bacc_deb_k = balanced_accuracy_score(y_true[val_idx], p_deb[val_idx].argmax(axis=1))
    bacc_q8_k = balanced_accuracy_score(y_true[val_idx], p_q8[val_idx].argmax(axis=1))
    bacc_tok_k = balanced_accuracy_score(y_true[val_idx], p_tok[val_idx].argmax(axis=1))
    print(f"  Fold {k}: DeBERTa={bacc_deb_k:.4f} | Qwen8={bacc_q8_k:.4f} | Blend={bacc_tok_k:.4f}")

bacc_total = balanced_accuracy_score(y_true, p_tok.argmax(axis=1))
print(f"\nOverall Out-Of-Fold Balanced Accuracy: {bacc_total:.4f}")

