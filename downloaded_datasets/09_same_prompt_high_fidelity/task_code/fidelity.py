"""Fidelity check: train a char+word TF-IDF logistic regression on competition train only, predict my passages.

Reports accuracy restricted to the labels present in my data, next to grouped CV accuracy on train itself.
Usage: fidelity.py <train.csv> <passages.csv>
"""
import sys

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tr = pd.read_csv(sys.argv[1])
ps = pd.read_csv(sys.argv[2])
ps = ps[ps.passage_ok]
cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000)
wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X = hstack([cv.fit_transform(tr.text), wv.fit_transform(tr.text)]).tocsr()
clf = LogisticRegression(C=10, max_iter=2000).fit(X, tr.model)
Xp = hstack([cv.transform(ps.text), wv.transform(ps.text)]).tocsr()
P = clf.predict_proba(Xp)
classes = list(clf.classes_)
present = sorted(ps.label.unique())
idx = [classes.index(c) for c in present]
pred_all = np.array(classes)[P.argmax(1)]
pred_sub = np.array(present)[P[:, idx].argmax(1)]
ps = ps.assign(pred_all=pred_all, pred_sub=pred_sub)
print("accuracy over all 18 labels:", round((ps.pred_all == ps.label).mean(), 3))
print(f"accuracy over the {len(present)} labels present (chance {1/len(present):.3f}):", round((ps.pred_sub == ps.label).mean(), 3))
ps["hit18"] = ps.pred_all == ps.label
ps["hit_sub"] = ps.pred_sub == ps.label
print(ps.groupby("label")[["hit18", "hit_sub"]].mean().round(2).to_string())
