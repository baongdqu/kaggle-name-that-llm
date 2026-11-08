import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from scipy.optimize import linear_sum_assignment

print("Testing V94 offline simulation...")
data = np.load('scratch/cache_cv_data.npz', allow_pickle=True)
base_logits = data['base_logits']
y_true = data['y_true']
all_pids = data['all_pids']
unique_pids = data['unique_pids']
LABELS = list(data['labels'])
L2I = {l: i for i, l in enumerate(LABELS)}

# Baseline 1: Pure independent argmax (V92 behavior on unclustered passages)
pred_argmax = base_logits.argmax(axis=1)
acc_argmax = balanced_accuracy_score(y_true, pred_argmax)
print(f"1. Independent argmax (V92 baseline): {acc_argmax*100:.2f}%")

# Baseline 2: Oracle Hungarian on True Prompts
pred_oracle = np.zeros_like(pred_argmax)
for pid in unique_pids:
    idx = np.where(all_pids == pid)[0]
    r, c = linear_sum_assignment(-base_logits[idx])
    pred_oracle[idx[r]] = c
acc_oracle = balanced_accuracy_score(y_true, pred_oracle)
print(f"2. Oracle Hungarian on True Prompts: {acc_oracle*100:.2f}%")

# Test 3: Unsupervised prompt clustering with character n-grams + TF-IDF (simulating test.csv)
from sklearn.feature_extraction.text import TfidfVectorizer

train_df = pd.read_csv('data/train.csv')
v_char = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=30000, sublinear_tf=True)
Xc_char = v_char.fit_transform(train_df['text'].values)

# Let's test Connected Components / cosine threshold clustering
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sim_matrix = (Xc_char @ Xc_char.T)

for threshold in [0.70, 0.75, 0.80, 0.85]:
    adj = (sim_matrix >= threshold).astype(int)
    n_components, labels = connected_components(csgraph=adj, directed=False)
    
    # Cap cluster sizes at 18
    unique_labs, counts = np.unique(labels, return_counts=True)
    n_multi = (counts >= 2).sum()
    n_capped = ((counts >= 2) & (counts <= 18)).sum()
    n_oversized = (counts > 18).sum()
    
    # Run Hungarian only on valid clusters of size [2, 18], fall back to argmax for singletons and oversized
    pred_sim = pred_argmax.copy()
    for lab in unique_labs:
        idx = np.where(labels == lab)[0]
        if 2 <= len(idx) <= 18:
            r, c = linear_sum_assignment(-base_logits[idx])
            pred_sim[idx[r]] = c
            
    acc_sim = balanced_accuracy_score(y_true, pred_sim)
    print(f"Threshold {threshold:.2f}: {acc_sim*100:.2f}% | Multi-item clusters: {n_multi} (Valid [2,18]: {n_capped}, Oversized: {n_oversized})")
