import numpy as np, pandas as pd, time
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('data/train.csv')
v1 = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=30000, sublinear_tf=True)
v2 = TfidfVectorizer(ngram_range=(1, 3), analyzer='word', max_features=20000, sublinear_tf=True)

t0 = time.time()
X1 = v1.fit_transform(train['text'])
X2 = v2.fit_transform(train['text'])
X = hstack([X1, X2], format='csr')
y = pd.factorize(train['model'])[0]
print(f"Features: {X.shape} created in {time.time()-t0:.2f}s")

t0 = time.time()
svc = LinearSVC(C=0.5, random_state=42, dual=False)
svc.fit(X, y)
print(f"LinearSVC (dual=False) took: {time.time()-t0:.2f}s")

t0 = time.time()
svc_d = LinearSVC(C=0.5, random_state=42, dual=True)
svc_d.fit(X, y)
print(f"LinearSVC (dual=True) took: {time.time()-t0:.2f}s")

