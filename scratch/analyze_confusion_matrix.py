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

print("=== CONFUSION MATRIX ANALYSIS (18 CLASSES) ===")
print("Overall Balanced Accuracy:", (np.diag(cm) / cm.sum(axis=1)).mean())

# Identify twin pairs
twin_pairs = [
    ('gpt-4o', 'gpt-4o-mini'),
    ('claude-3-5-sonnet', 'claude-3-haiku'),
    ('gemini-1-5-pro', 'gemini-1-5-flash'),
    ('qwen-2-5-72b', 'qwen-2-5-7b'),
    ('gemma-2-9b', 'gemma-2-27b'),
    ('deepseek-coder-v2', 'deepseek-v2-5'),
    ('llama-3-1-8b', 'llama-3-1-70b'),
]

# Map to actual labels in dataset
label_set = set(labels)
print("\nActual labels in train:", labels)

found_twins = []
for m1, m2 in twin_pairs:
    # Match closest in labels
    match1 = [l for l in labels if m1.replace('-', '').lower() in l.replace('-', '').lower() or l.startswith(m1[:6])]
    match2 = [l for l in labels if m2.replace('-', '').lower() in l.replace('-', '').lower() or l.startswith(m2[:6])]
    if match1 and match2:
        found_twins.append((match1[0], match2[0]))

print("\nIdentified Twin Pairs:")
total_twin_errors = 0
total_errors = (y_true != preds).sum()

for c1, c2 in found_twins:
    i1, i2 = l2i[c1], l2i[c2]
    e12 = cm[i1, i2] # true c1 predicted as c2
    e21 = cm[i2, i1] # true c2 predicted as c1
    acc1 = cm[i1, i1] / cm[i1].sum()
    acc2 = cm[i2, i2] / cm[i2].sum()
    total_twin_errors += (e12 + e21)
    print(f"Twin: {c1:20s} ({acc1*100:.1f}%) <-> {c2:20s} ({acc2*100:.1f}%) | Cross-Errors: {e12} + {e21} = {e12+e21}")

print(f"\nTotal Errors: {total_errors}")
print(f"Errors within Twin Pairs: {total_twin_errors} ({total_twin_errors/total_errors*100:.1f}% of ALL errors!)")

