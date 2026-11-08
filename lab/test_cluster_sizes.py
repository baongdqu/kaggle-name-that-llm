import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, fcluster

train = pd.read_csv('data/train.csv')
print(f"Total train rows: {len(train)}, total unique prompt_ids: {train.prompt_id.nunique()}")

# Simulate test clustering logic from infer.py
vec = TfidfVectorizer(token_pattern=r'(?u)\b[a-zA-Z]{3,}\b', min_df=2, max_df=0.8, stop_words='english')
X = vec.fit_transform(train.text).toarray()
sim = X @ X.T
dist = 1.0 - sim
np.fill_diagonal(dist, 0.0)
dist = np.clip(dist, 0.0, None)
from scipy.spatial.distance import squareform
dist_condensed = squareform(dist, checks=False)

Z = linkage(dist_condensed, method='average')
adapt_thr = 0.15
labels = fcluster(Z, t=1.0 - adapt_thr, criterion='distance')
sizes = pd.Series(labels).value_counts()
print(f"Number of estimated clusters: {len(sizes)}")
print(f"Clusters >= 12: {(sizes >= 12).sum()} clusters, covering {(sizes[sizes >= 12]).sum()} rows")
print(f"Clusters < 12: {(sizes < 12).sum()} clusters, covering {(sizes[sizes < 12]).sum()} rows")
print("Top 10 cluster sizes:", sizes.head(10).tolist())
print("Distribution of cluster sizes:\n", pd.cut(sizes, bins=[0, 1, 3, 6, 11, 18, 50]).value_counts())

