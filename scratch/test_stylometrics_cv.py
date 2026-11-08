import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

train = pd.read_csv('data/train.csv')

families = {
    'opus': ['claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5'],
    'gpt56': ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra'],
    'gemini_flash': ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash']
}

for fam, sabs in families.items():
    sub = train[train['model'].isin(sabs)].reset_index(drop=True)
    label_map = {m: i for i, m in enumerate(sabs)}
    y_int = np.array([label_map[m] for m in sub['model'].values])
    
    feats = []
    for t in sub['text'].values:
        words = str(t).split()
        wlen = len(words)
        clen = len(str(t))
        feats.append([
            wlen,
            clen / max(1, wlen),
            str(t).count('—'),
            str(t).count(';'),
            str(t).count(':'),
            str(t).count('?'),
            str(t).count('!'),
            str(t).count('('),
            str(t).count('"')
        ])
    X = np.array(feats, dtype=np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr, val in skf.split(X, y_int):
        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(X[tr], y_int[tr])
        preds = clf.predict(X[val])
        scores.append(balanced_accuracy_score(y_int[val], preds))
    print(f'Family {fam:12s} 9-stylometrics CV Balanced Accuracy: {np.mean(scores)*100:.2f}% (random={100/3:.1f}%)')
