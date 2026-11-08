import pandas as pd, numpy as np, re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

train = pd.read_csv('data/train.csv')

def evaluate_pair(m1, m2, specific_puncs):
    print("\n" + "="*60)
    print(f"🔬 Testing Optimization on: {m1} vs {m2}")
    print("="*60)
    
    sub = train[train['model'].isin([m1, m2])].copy().reset_index(drop=True)
    pids = sub['prompt_id'].values
    upids = np.unique(pids)
    y = (sub['model'] == m2).astype(int).values
    texts = sub['text'].values
    
    wlens = np.array([len(t.split()) for t in texts], dtype=np.float32)
    
    # 1. Standard Char N-Grams baseline
    v1 = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=15000, sublinear_tf=True)
    X1 = v1.fit_transform(texts)
    
    # 2. Optimized char_wb (4-6) + Prompt-relative Z-Scores
    v2 = TfidfVectorizer(ngram_range=(4, 6), analyzer='char_wb', max_features=25000, sublinear_tf=True)
    
    z_feats = []
    # Length z-score
    z_w = np.zeros_like(wlens)
    for p in upids:
        idx = np.where(pids == p)[0]
        m_ = wlens[idx].mean()
        s_ = max(1.0, wlens[idx].std())
        z_w[idx] = (wlens[idx] - m_) / s_
    z_feats.append(z_w)
    
    # Punctuation z-scores & flags
    for punc in specific_puncs:
        cnts = np.array([t.count(punc) for t in texts], dtype=np.float32)
        z_c = np.zeros_like(cnts)
        for p in upids:
            idx = np.where(pids == p)[0]
            m_ = cnts[idx].mean()
            s_ = max(1.0, cnts[idx].std())
            z_c[idx] = (cnts[idx] - m_) / s_
        z_feats.append(z_c)
        z_feats.append((cnts > 0).astype(np.float32))
        
    dense_z = np.column_stack(z_feats)
    X2 = hstack([v2.fit_transform(texts), csr_matrix(dense_z * 2.5)], format='csr')
    
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
    
    print(f"📊 Standard Baseline:               {acc1*100:.2f}%")
    print(f"🚀 Optimized (char_wb + Z-Scores):   {acc2*100:.2f}% (Gain: {acc2*100 - acc1*100:+.2f}%) 🔥")

# Test GPT-5.6 Sol vs Terra (curly quotes, parentheses, question marks, exclamation)
evaluate_pair('gpt-5.6-sol', 'gpt-5.6-terra', ['“', '”', '(', ')', '?', '!'])

# Test GLM-5.3 vs Flash (parentheses, slash, exclamation)
evaluate_pair('glm-5.3', 'glm-5.3-flash', ['(', ')', '/', '!'])
