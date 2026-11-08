import numpy as np, pandas as pd, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier

train = pd.read_csv('data/train.csv')
v = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=30000, sublinear_tf=True)
X = v.fit_transform(train['text'])
y = pd.factorize(train['model'])[0]
print(f"TFIDF shape: {X.shape}")

t0 = time.time()
svc = LinearSVC(C=0.5, random_state=42)
svc.fit(X, y)
print(f"LinearSVC took: {time.time()-t0:.2f}s")

t0 = time.time()
ridge = RidgeClassifier(alpha=1.0, random_state=42)
ridge.fit(X, y)
print(f"RidgeClassifier took: {time.time()-t0:.2f}s")

