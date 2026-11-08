import numpy as np, pandas as pd, re
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train = pd.read_csv('data/train.csv')

def analyze_pair(m1, m2):
    sub = train[train['model'].isin([m1, m2])].copy().reset_index(drop=True)
    y = (sub['model'] == m2).astype(int).values
    groups = sub['prompt_id'].values
    texts = sub['text'].values
    
    # Try different TF-IDF configurations
    for analyzer, ngram, mf in [
        ('word', (1, 2), 5000),
        ('char', (3, 5), 10000),
        ('char_wb', (2, 5), 10000),
    ]:
        vec = TfidfVectorizer(ngram_range=ngram, analyzer=analyzer, min_df=2, max_features=mf, sublinear_tf=True)
        X = vec.fit_transform(texts)
        
        gkf = GroupKFold(n_splits=5)
        oof = np.zeros(len(sub))
        for tr, va in gkf.split(X, y, groups):
            clf = LogisticRegression(C=1.0, max_iter=500)
            clf.fit(X[tr], y[tr])
            oof[va] = clf.predict_proba(X[va])[:, 1]
            
        acc = accuracy_score(y, oof > 0.5)
        print(f"[{m1} vs {m2}] {analyzer} {ngram}: CV Acc = {acc*100:.2f}%")

print("--- Sol vs Terra ---")
analyze_pair('gpt-5.6-sol', 'gpt-5.6-terra')

print("\n--- GLM vs GLM-Flash ---")
analyze_pair('glm-5.3', 'glm-5.3-flash')

print("\n--- Luna vs Sol ---")
analyze_pair('gpt-5.6-luna', 'gpt-5.6-sol')

print("\n--- Gemini 3.7 vs 3.8 ---")
analyze_pair('gemini-3.7-flash', 'gemini-3.8-flash')

