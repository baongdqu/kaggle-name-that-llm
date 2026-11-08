import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

train = pd.read_csv('data/train.csv')
print(f"Loaded data/train.csv: {len(train)} rows")

clean = [t.lower() for t in train.text]
vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=list(ENGLISH_STOP_WORDS), max_features=10000)
X = vec.fit_transform(clean)
sim = (X @ X.T).toarray()
dist = np.clip(1.0 - sim, 0.0, 1.0)
np.fill_diagonal(dist, 0.0)
Z = linkage(squareform(dist, checks=False), method="average")

for thresh in [0.75, 0.80, 0.85, 0.90]:
    labels = fcluster(Z, t=thresh, criterion="distance")
    counts = pd.Series(labels).value_counts()
    sizes = counts[counts >= 2]
    size_2 = (counts == 2).sum()
    size_3 = (counts == 3).sum()
    size_4plus = (counts >= 4).sum()
    print(f"\nThresh {thresh}: total clusters >= 2: {len(sizes)}, total items in clusters: {sizes.sum()}")
    print(f"  size == 2: {size_2} clusters ({size_2*2} items)")
    print(f"  size == 3: {size_3} clusters ({size_3*3} items)")
    print(f"  size >= 4: {size_4plus} clusters ({sizes[counts >= 4].sum()} items)")
