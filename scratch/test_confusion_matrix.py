import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix
import test_cross_72_milestone as m

cm = confusion_matrix(m.y_true, m.best_preds)
LABELS = m.LABELS

confused = []
for i in range(18):
    for j in range(18):
        if i != j and cm[i, j] > 0:
            confused.append((cm[i, j], LABELS[i], LABELS[j]))

confused.sort(reverse=True)
print("\n=== TOP 25 CONFUSED PAIRS (True -> Predicted) ===")
for cnt, t, p in confused[:25]:
    print(f"{cnt:3d} errors: True [{t:22s}] -> Predicted as [{p:22s}]")

