import numpy as np, pandas as pd, re
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import lru_cache

train = pd.read_csv('data/train.csv')
pids = train['prompt_id'].unique()

def evaluate_pair_difference(m1_name, m2_name):
    print(f"\n--- Testing Pairwise Difference on {m1_name} vs {m2_name} ---", flush=True)
    sub = train[train['model'].isin([m1_name, m2_name])].copy()
    
    # We test with TF-IDF word 1-2
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True)
    X_tfidf = vec.fit_transform(train['text']).toarray()
    
    pairs = []
    y_pairs = []
    pair_pids = []
    
    for p in pids:
        rows = sub[sub['prompt_id'] == p]
        if len(rows) == 2:
            idx1 = rows.index[0]
            idx2 = rows.index[1]
            m1 = rows.loc[idx1, 'model']
            m2 = rows.loc[idx2, 'model']
            if m1 != m2:
                f1 = X_tfidf[idx1]
                f2 = X_tfidf[idx2]
                if m1 == m2_name:
                    pairs.append(f1 - f2)
                    y_pairs.append(1)
                    pairs.append(f2 - f1)
                    y_pairs.append(0)
                else:
                    pairs.append(f2 - f1)
                    y_pairs.append(1)
                    pairs.append(f1 - f2)
                    y_pairs.append(0)
                pair_pids.append(p)
                pair_pids.append(p)

    X_diff = np.array(pairs)
    y_diff = np.array(y_pairs)
    pair_pids = np.array(pair_pids)
    
    gkf = GroupKFold(n_splits=5)
    for C in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
        oof = np.zeros(len(y_diff))
        for tr, va in gkf.split(X_diff, y_diff, groups=pair_pids):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500))
            clf.fit(X_diff[tr], y_diff[tr])
            oof[va] = clf.predict_proba(X_diff[va])[:, 1]
        
        pos_idx = np.arange(0, len(y_diff), 2)
        correct = (oof[pos_idx] > oof[pos_idx + 1]).mean()
        print(f"C={C}: Accuracy = {correct*100:.2f}%")

evaluate_pair_difference('gpt-5.6-sol', 'gpt-5.6-terra')
evaluate_pair_difference('gemini-3.7-flash', 'gemini-3.8-flash')
evaluate_pair_difference('claude-opus-4-6', 'claude-opus-4-8')
