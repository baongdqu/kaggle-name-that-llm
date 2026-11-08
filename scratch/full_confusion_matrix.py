import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

train = pd.read_csv('data/train.csv')
labels = sorted(train.model.unique())
l2i = {l: i for i, l in enumerate(labels)}
y_true = np.array([l2i[m] for m in train.model.values])

cache = np.load('lab/cache/base_oof.npz')
base_probs = cache['probs']
preds = base_probs.argmax(axis=1)

cm = confusion_matrix(y_true, preds)

# Define families
families = {
    'Claude': [l for l in labels if l.startswith('claude')],
    'Gemini': [l for l in labels if l.startswith('gemini')],
    'GPT': [l for l in labels if l.startswith('gpt')],
    'GLM': [l for l in labels if l.startswith('glm')],
    'Singletons': ['kimi-k3', 'qwen3.8-max']
}

print("=== FAMILY-WISE ERROR BREAKDOWN ===", flush=True)
total_errors = (y_true != preds).sum()
intra_family_errors = 0
inter_family_errors = 0

for fam_name, members in families.items():
    if len(members) <= 1:
        continue
    fam_idx = [l2i[m] for m in members]
    fam_err = 0
    for i in fam_idx:
        for j in fam_idx:
            if i != j:
                fam_err += cm[i, j]
    intra_family_errors += fam_err
    print(f"Family {fam_name:8s} ({len(members)} models): Intra-family confusion = {fam_err} errors", flush=True)

inter_family_errors = total_errors - intra_family_errors
print(f"\nTotal Errors: {total_errors}", flush=True)
print(f"Intra-Family (Sibling Confusion): {intra_family_errors} ({intra_family_errors/total_errors*100:.1f}%)", flush=True)
print(f"Inter-Family (Cross-Brand Errors): {inter_family_errors} ({inter_family_errors/total_errors*100:.1f}%)", flush=True)

# Top 15 specific model confusion pairs
print("\n=== TOP 15 CONFUSION PAIRS ===", flush=True)
pairs = []
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        c_ij = cm[i, j] + cm[j, i]
        pairs.append((c_ij, labels[i], labels[j], cm[i, j], cm[j, i]))

pairs.sort(key=lambda x: x[0], reverse=True)
for count, m1, m2, e12, e21 in pairs[:15]:
    same_fam = (m1.split('-')[0] == m2.split('-')[0])
    fam_tag = "[SIBLING]" if same_fam else "[CROSS-BRAND]"
    print(f"{fam_tag:13s} {m1:22s} <-> {m2:22s} | Total: {count:3d} ({m1}->{m2}: {e12}, {m2}->{m1}: {e21})", flush=True)

