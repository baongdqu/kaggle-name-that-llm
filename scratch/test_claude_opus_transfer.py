import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

train = pd.read_csv('data/train.csv')
claude_ext = pd.read_csv('data/ntl_extra_passages_claude.csv')

models = ['claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5']

# Train on ntl_extra_passages_claude
sub_ext = claude_ext[claude_ext['model'].isin(models)].copy().reset_index(drop=True)
X_train_text = sub_ext['text'].values
y_train = np.array([models.index(m) for m in sub_ext['model'].values])

# Test on official train.csv
sub_val = train[train['model'].isin(models)].copy().reset_index(drop=True)
X_val_text = sub_val['text'].values
y_val = np.array([models.index(m) for m in sub_val['model'].values])

v1 = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=35000, sublinear_tf=True)
v2 = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', max_features=25000, sublinear_tf=True)
from scipy.sparse import hstack
X_tr = hstack([v1.fit_transform(X_train_text), v2.fit_transform(X_train_text)], format='csr')
X_va = hstack([v1.transform(X_val_text), v2.transform(X_val_text)], format='csr')

svc = LinearSVC(C=0.5, random_state=42, max_iter=2000)
svc.fit(X_tr, y_train)
preds = svc.predict(X_va)

acc = accuracy_score(y_val, preds)
print(f"Total external train: {len(sub_ext)} | Total official val: {len(sub_val)}")
print(f"🔥 Zero-Shot Transfer Accuracy on 3 Opus Siblings: {acc*100:.2f}% (vs 33.33% random guessing)!")

