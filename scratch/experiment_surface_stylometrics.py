import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

df = pd.read_csv('data/train.csv')
print('Loaded train.csv:', df.shape)

# Let us extract Character 3-5 gram TF-IDF
char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=15000, sublinear_tf=True)
X_char = char_vec.fit_transform(df['text'])

word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
X_word = word_vec.fit_transform(df['text'])

from scipy.sparse import hstack
X_all = hstack([X_char, X_word]).tocsr()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(df), dtype=int)
classes = np.sort(df['model'].unique())
c2i = {c: i for i, c in enumerate(classes)}
y = df['model'].map(c2i).values

accs = []
for fold, (train_idx, val_idx) in enumerate(skf.split(df, y)):
    clf = LogisticRegression(C=2.0, max_iter=300, solver='lbfgs')
    clf.fit(X_all[train_idx], y[train_idx])
    val_pred = clf.predict(X_all[val_idx])
    acc = balanced_accuracy_score(y[val_idx], val_pred)
    accs.append(acc)
    print(f'Fold {fold} Surface Stylometric Acc: {acc:.4f}')

print(f'Mean CV Surface Stylometric Acc: {np.mean(accs):.4f}')
