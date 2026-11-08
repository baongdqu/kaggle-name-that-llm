import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix

# Let's inspect the confusion matrix from scratch/test_cross_72_milestone.py
# Run the evaluation and print top 15 confused pairs
import subprocess
res = subprocess.run(['python', '-c', '''
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix
# import and run test_cross_72_milestone
import test_cross_72_milestone as m
cm = confusion_matrix(m.y_true, m.best_preds)
LABELS = m.LABELS

confused = []
for i in range(18):
    for j in range(18):
        if i != j and cm[i, j] > 0:
            confused.append((cm[i, j], LABELS[i], LABELS[j]))

confused.sort(reverse=True)
print("=== TOP 20 CONFUSED PAIRS (True -> Predicted) ===")
for cnt, t, p in confused[:20]:
    print(f"{cnt:3d} errors: True [{t}] was predicted as [{p}]")
'''], capture_output=True, text=True, cwd='scratch')

print(res.stdout)
if res.stderr:
    print("ERR:", res.stderr[-500:])

