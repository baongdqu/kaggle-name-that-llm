import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster

train = pd.read_csv('data/train.csv')
texts = train['text'].astype(str).tolist()
pids = train['prompt_id'].values
unique_pids = np.unique(pids)
y_true = train['model'].map(lambda x: sorted(train['model'].unique()).index(x)).values

print("--- 1. Testing Unsupervised Prompt Clustering on train.csv (as if it were test.csv) ---", flush=True)

# Use TF-IDF content words (exclude rare and common words)
vec = TfidfVectorizer(max_features=25000, stop_words='english', sublinear_tf=True, min_df=2, max_df=0.2)
X = vec.fit_transform(texts)
sim = (X @ X.T).toarray()

# Agglomerative clustering into 150 clusters
# Using cosine distance
dist = 1.0 - sim
np.fill_diagonal(dist, 0.0)
dist = np.clip(dist, 0.0, 2.0)

# Condensed distance matrix for scipy linkage
from scipy.spatial.distance import squareform
condensed_dist = squareform(dist, checks=False)

Z = linkage(condensed_dist, method='average')
pred_clusters = fcluster(Z, t=150, criterion='maxclust')

ari = adjusted_rand_score(pids, pred_clusters)
print(f"Adjusted Rand Index of Reconstructed Clusters: {ari*100:.2f}% (100% is perfect prompt recovery)")

# Check cluster size distribution
sizes = pd.Series(pred_clusters).value_counts()
print(f"Cluster sizes: Min={sizes.min()}, Max={sizes.max()}, Mean={sizes.mean():.1f}, Median={sizes.median()}")

# Now evaluate Hungarian matching using PREDICTED clusters vs TRUE clusters!
data = np.load('scratch/cache_cv_data.npz', allow_pickle=True)
base_logits = data['base_logits']

# 1. Using TRUE clusters (our 72.73% CV setup):
preds_true_clusters = np.zeros(len(y_true), dtype=int)
for p in unique_pids:
    idx = np.where(pids == p)[0]
    M = base_logits[idx]
    r, c = linear_sum_assignment(-M)
    preds_true_clusters[idx[r]] = c
acc_true = balanced_accuracy_score(y_true, preds_true_clusters)
print(f"\nHungarian with TRUE Prompts: {acc_true*100:.2f}%")

# 2. Using PREDICTED clusters (without knowing prompt_id, exactly as in test.csv!):
preds_pred_clusters = np.zeros(len(y_true), dtype=int)
for c_id in np.unique(pred_clusters):
    idx = np.where(pred_clusters == c_id)[0]
    M = base_logits[idx]
    if len(idx) <= 18:
        # Pad or solve directly
        r, c = linear_sum_assignment(-M)
        preds_pred_clusters[idx[r]] = c
    else:
        # If cluster > 18, take argmax or split
        preds_pred_clusters[idx] = M.argmax(axis=1)

acc_pred = balanced_accuracy_score(y_true, preds_pred_clusters)
print(f"Hungarian with PREDICTED Prompts: {acc_pred*100:.2f}%")

# 3. Pure Argmax (perpassage=1):
preds_argmax = base_logits.argmax(axis=1)
acc_argmax = balanced_accuracy_score(y_true, preds_argmax)
print(f"Pure Independent Argmax (perpassage=1): {acc_argmax*100:.2f}%")

