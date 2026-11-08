import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
from scipy.sparse import hstack

train = pd.read_csv('data/train.csv')
LABELS = sorted(train.model.unique())
L2I = {l: i for i, l in enumerate(LABELS)}
y = train.model.map(L2I).values
groups = train.prompt_id.values

_marks = [' — ', '—', '–', '’', '‘', '“', '”', '«', '»', '"', ';', ':', '(', ')', '[', ']', '**', '`', '…', '•', '\n', '\n\n', 'e.g.', 'i.e.', 'However,', 'Overall,', 'In conclusion', 'Therefore,', 'Moreover,', '\t', ' - ', ' = ', '\n- ', '\n1.', '###', '```', '> ']
X_markers = np.array([[float(m in t) for m in _marks] for t in train.text.values], dtype=np.float32)

FAMILIES = {
    'gemini': [i for i, l in enumerate(LABELS) if l.startswith('gemini')],
    'glm': [i for i, l in enumerate(LABELS) if l.startswith('glm')],
    'claude': [i for i, l in enumerate(LABELS) if l.startswith('claude')],
    'gpt': [i for i, l in enumerate(LABELS) if l.startswith('gpt')]
}

gkf = GroupKFold(n_splits=5)
for fam, idx in FAMILIES.items():
    mask = np.isin(y, idx)
    y_fam = y[mask]
    lbl_map = {c: i for i, c in enumerate(idx)}
    y_loc = np.array([lbl_map[c] for c in y_fam])
    texts = train.text.values[mask]
    grps = groups[mask]
    markers = X_markers[mask]
    
    oof = np.zeros((len(y_loc), len(idx)))
    for tr, val in gkf.split(texts, y_loc, grps):
        vec = FeatureUnion([
            ('word', TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
            ('char', TfidfVectorizer(ngram_range=(2, 5), analyzer='char_wb', max_features=12000, sublinear_tf=True))
        ])
        X_tr_tfidf = vec.fit_transform(texts[tr])
        X_val_tfidf = vec.transform(texts[val])
        
        # Stack TF-IDF with 37 Typography stylometric markers!
        X_tr = hstack([X_tr_tfidf, markers[tr]])
        X_val = hstack([X_val_tfidf, markers[val]])
        
        clf = LogisticRegression(C=0.5, max_iter=300, random_state=42)
        clf.fit(X_tr, y_loc[tr])
        oof[val] = clf.predict_proba(X_val)
    
    oof_pred = oof.argmax(axis=1)
    acc = balanced_accuracy_score(y_loc, oof_pred)
    print(f'Family {fam:8s}: Sibling Gate + 37 Typography Markers OOF Balanced Acc = {acc*100:.2f}% (Chance = {100.0/len(idx):.1f}%)')

