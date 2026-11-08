import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

df = pd.read_csv('data/train.csv')

# Test Sibling sub-classification: gemini-3.6-flash vs gemini-3.7-flash vs gemini-3.8-flash
gemini_df = df[df['model'].str.contains('gemini-3.[678]-flash')].copy()
print('Gemini flash samples:', len(gemini_df))

char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=10000, sublinear_tf=True)
X_char = char_vec.fit_transform(gemini_df['text'])

y = gemini_df['model'].values
classes = np.unique(y)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accs = []
for fold, (train_idx, val_idx) in enumerate(skf.split(gemini_df, y)):
    clf = LogisticRegression(C=1.0, max_iter=200)
    clf.fit(X_char[train_idx], y[train_idx])
    preds = clf.predict(X_char[val_idx])
    acc = balanced_accuracy_score(y[val_idx], preds)
    accs.append(acc)

print(f'Gemini Flash Sibling Separation Acc: {np.mean(accs):.4f} (Random chance is 0.3333)')
