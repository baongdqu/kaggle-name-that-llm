import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from scipy.special import logsumexp, softmax

train = pd.read_csv('data/train.csv')
print('Train shape:', train.shape)

LABELS = sorted(train.model.unique())
L2I = {l: i for i, l in enumerate(LABELS)}
y = train.model.map(L2I).values

FAMILY_IDX = {f: np.array([i for i, l in enumerate(LABELS) if l.startswith(f)]) for f in ('claude', 'gpt', 'gemini', 'glm')}

print('Family counts in train:')
for fam, idx in FAMILY_IDX.items():
    print(f"{fam} ({len(idx)} models): {[LABELS[i] for i in idx]}")

# Let's test fitting a specialized family gate for Gemini and GLM on character & word n-grams
for fam, idx in FAMILY_IDX.items():
    mask = np.isin(y, idx)
    y_fam = y[mask]
    # map to local class indices 0..len(idx)-1
    lbl_to_local = {c: i for i, c in enumerate(idx)}
    y_local = np.array([lbl_to_local[c] for c in y_fam])
    texts = train.text.values[mask]
    
    vec = TfidfVectorizer(ngram_range=(1, 3), max_features=15000, sublinear_tf=True)
    X = vec.fit_transform(texts)
    clf = LinearSVC(C=0.5, random_state=42)
    clf.fit(X, y_local)
    
    acc = clf.score(X, y_local)
    print(f"Family {fam}: fitted {len(y_local)} samples, train accuracy = {acc:.4f}")

print("Family gate test completed successfully!")

