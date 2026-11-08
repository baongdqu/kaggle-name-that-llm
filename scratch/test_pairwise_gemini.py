import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

train = pd.read_csv('data/train.csv')

# Filter only gemini-3.7-flash and gemini-3.8-flash
sub = train[train['model'].isin(['gemini-3.7-flash', 'gemini-3.8-flash'])].copy().reset_index(drop=True)
print(f"Total samples for Gemini 3.7 vs 3.8: {len(sub)}")

y = (sub['model'] == 'gemini-3.8-flash').astype(int).values
texts = sub['text'].values
pids = sub['prompt_id'].values
unique_pids = np.unique(pids)

# Test multiple feature sets:
for feat_type in ['char_3_5', 'char_2_4', 'word_1_3', 'word_char_combo']:
    if feat_type == 'char_3_5':
        v = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', sublinear_tf=True, min_df=1)
        X = v.fit_transform(texts)
    elif feat_type == 'char_2_4':
        v = TfidfVectorizer(ngram_range=(2, 4), analyzer='char', sublinear_tf=True, min_df=1)
        X = v.fit_transform(texts)
    elif feat_type == 'word_1_3':
        v = TfidfVectorizer(ngram_range=(1, 3), analyzer='word', sublinear_tf=True, min_df=1)
        X = v.fit_transform(texts)
    elif feat_type == 'word_char_combo':
        v1 = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', sublinear_tf=True, min_df=1)
        v2 = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', sublinear_tf=True, min_df=1)
        from scipy.sparse import hstack
        X = hstack([v1.fit_transform(texts), v2.fit_transform(texts)], format='csr')

    # 5-fold CV grouped by prompt_id
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(len(sub))
    for tr_pid_idx, val_pid_idx in kf.split(unique_pids):
        tr_pids = unique_pids[tr_pid_idx]
        val_pids = unique_pids[val_pid_idx]
        tr_idx = np.where(np.isin(pids, tr_pids))[0]
        val_idx = np.where(np.isin(pids, val_pids))[0]
        
        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(X[tr_idx], y[tr_idx])
        preds[val_idx] = clf.predict_proba(X[val_idx])[:, 1]

    acc = accuracy_score(y, (preds >= 0.5).astype(int))
    print(f"[{feat_type}] Binary CV Accuracy (Prompt-Grouped): {acc*100:.2f}% (vs random 50.0%)")

