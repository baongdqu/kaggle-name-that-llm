import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

train = pd.read_csv('data/train.csv')
texts = train['text'].astype(str).tolist()
pids = train['prompt_id'].values

def test_pair(name1, name2):
    sub = train[train['model'].isin([name1, name2])].copy().reset_index(drop=True)
    y = (sub['model'] == name2).astype(int).values
    grp = sub['prompt_id'].values
    sub_texts = sub['text'].tolist()
    
    print(f"\n--- Testing Pair: [{name1}] vs [{name2}] (Total samples: {len(sub)}) ---")
    
    # 1. Word TF-IDF
    v_w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10000, sublinear_tf=True)
    X_w = v_w.fit_transform(sub_texts)
    
    gkf = GroupKFold(n_splits=5)
    for C in [0.1, 0.5, 1.0, 2.0]:
        oof = np.zeros(len(sub))
        for tr, va in gkf.split(X_w, y, groups=grp):
            clf = LogisticRegression(C=C, max_iter=500)
            clf.fit(X_w[tr], y[tr])
            oof[va] = clf.predict_proba(X_w[va])[:, 1]
        print(f"  Word (1,2) C={C}: Accuracy = {accuracy_score(y, oof > 0.5)*100:.2f}%")
        
    # 2. Char TF-IDF
    v_c = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', min_df=2, max_features=15000, sublinear_tf=True)
    X_c = v_c.fit_transform(sub_texts)
    for C in [0.1, 0.5, 1.0, 2.0]:
        oof = np.zeros(len(sub))
        for tr, va in gkf.split(X_c, y, groups=grp):
            clf = LogisticRegression(C=C, max_iter=500)
            clf.fit(X_c[tr], y[tr])
            oof[va] = clf.predict_proba(X_c[va])[:, 1]
        print(f"  Char (3,5) C={C}: Accuracy = {accuracy_score(y, oof > 0.5)*100:.2f}%")

test_pair('claude-sonnet-5', 'claude-opus-4-8')
test_pair('kimi-k3', 'gpt-5.5')
test_pair('gpt-5.6-sol', 'gpt-5.6-terra')

