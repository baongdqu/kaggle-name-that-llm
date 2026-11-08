import pandas as pd, numpy as np, re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

train = pd.read_csv('data/train.csv')
gemini_models = ['gemini-3.7-flash', 'gemini-3.8-flash']
sub = train[train['model'].isin(gemini_models)].copy().reset_index(drop=True)
pids = sub['prompt_id'].values
upids = np.unique(pids)
y = (sub['model'] == 'gemini-3.8-flash').astype(int).values
texts = sub['text'].values

# Standard features
wlens = np.array([len(t.split()) for t in texts], dtype=np.float32)
colons = np.array([t.count(':') for t in texts], dtype=np.float32)
hyphens = np.array([t.count('-') for t in texts], dtype=np.float32)

# Prompt-relative Z-score features
z_wlens = np.zeros_like(wlens)
z_hyphens = np.zeros_like(hyphens)

for p in upids:
    idx = np.where(pids == p)[0]
    # Length relative z-score
    m_w = wlens[idx].mean()
    s_w = max(1.0, wlens[idx].std())
    z_wlens[idx] = (wlens[idx] - m_w) / s_w
    # Hyphen relative z-score
    m_h = hyphens[idx].mean()
    s_h = max(1.0, hyphens[idx].std())
    z_hyphens[idx] = (hyphens[idx] - m_h) / s_h

# Model 1: char_wb (3, 5) without z-score
v1 = TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', max_features=15000, sublinear_tf=True)
X1 = v1.fit_transform(texts)

# Model 2: char_wb (4, 6) + prompt-relative z-score features
v2 = TfidfVectorizer(ngram_range=(4, 6), analyzer='char_wb', max_features=25000, sublinear_tf=True)
dense_rel = np.column_stack([z_wlens, z_hyphens, (wlens < 110).astype(np.float32)])
X2 = hstack([v2.fit_transform(texts), csr_matrix(dense_rel * 2.0)], format='csr')

kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds1 = np.zeros(len(sub))
preds2 = np.zeros(len(sub))

for tr_idx, va_idx in kf.split(upids):
    tr = np.where(np.isin(pids, upids[tr_idx]))[0]
    va = np.where(np.isin(pids, upids[va_idx]))[0]
    
    clf1 = LogisticRegression(C=1.0, max_iter=500)
    clf1.fit(X1[tr], y[tr])
    preds1[va] = clf1.predict_proba(X1[va])[:, 1]
    
    clf2 = LogisticRegression(C=1.0, max_iter=500)
    clf2.fit(X2[tr], y[tr])
    preds2[va] = clf2.predict_proba(X2[va])[:, 1]

acc1 = accuracy_score(y, preds1 > 0.5)
acc2 = accuracy_score(y, preds2 > 0.5)

print(f"📊 Standard Char N-Grams Baseline:              {acc1*100:.2f}%")
print(f"🚀 Optimized Char_wb (4-6) + Relative Z-Scores:  {acc2*100:.2f}% (Gain: {acc2*100 - acc1*100:+.2f}%)")
