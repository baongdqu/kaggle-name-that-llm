import pandas as pd, numpy as np, re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

train = pd.read_csv('data/train.csv')

def extract_forensic_features(df):
    texts = df['text'].astype(str).values
    wlens = np.array([len(t.split()) for t in texts], dtype=np.float32)
    
    feats = []
    # 1. Punctuation counts
    puncs = [':', ';', '—', '–', '-', '“', '”', '’', '‘', '(', ')', '/', '!', '?', '`', '*', '"', "'"]
    for p in puncs:
        c = np.array([t.count(p) for t in texts], dtype=np.float32)
        feats.append(c)
        feats.append(c / (wlens + 1.0)) # density
        feats.append((c > 0).astype(np.float32)) # binary presence
        
    # 2. Capitalization patterns
    caps = np.array([len(re.findall(r'[A-Z]', t)) for t in texts], dtype=np.float32)
    feats.append(caps / (wlens + 1.0))
    
    # 3. Sentence metrics
    sents = np.array([len(re.findall(r'[.!?]+', t)) for t in texts], dtype=np.float32)
    feats.append(sents)
    feats.append(wlens / (sents + 1.0)) # words per sentence
    
    # 4. Starting word indicators
    openings = ['the', 'in', 'beneath', 'deep', 'when', 'imagine', 'at', 'on', 'across']
    for op in openings:
        has_op = np.array([1.0 if t.lower().startswith(op) else 0.0 for t in texts], dtype=np.float32)
        feats.append(has_op)
        
    return np.column_stack(feats)

def test_pair_separation(m1, m2):
    sub = train[train['model'].isin([m1, m2])].copy().reset_index(drop=True)
    pids = sub['prompt_id'].values
    upids = np.unique(pids)
    y = (sub['model'] == m2).astype(int).values
    
    # Forensic dense features
    F_dense = extract_forensic_features(sub)
    
    # Text n-grams
    v_c = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=15000, sublinear_tf=True)
    v_w = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', max_features=10000, sublinear_tf=True)
    
    X_text = hstack([v_c.fit_transform(sub['text']), v_w.fit_transform(sub['text'])], format='csr')
    X_combo = hstack([X_text, csr_matrix(F_dense)], format='csr')
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds_combo = np.zeros(len(sub))
    preds_text = np.zeros(len(sub))
    
    for tr_p_idx, val_p_idx in kf.split(upids):
        tr_p = upids[tr_p_idx]
        val_p = upids[val_p_idx]
        tr_idx = np.where(np.isin(pids, tr_p))[0]
        val_idx = np.where(np.isin(pids, val_p))[0]
        
        # Text only
        clf1 = LogisticRegression(C=1.0, max_iter=500)
        clf1.fit(X_text[tr_idx], y[tr_idx])
        preds_text[val_idx] = clf1.predict_proba(X_text[val_idx])[:, 1]
        
        # Text + Forensic features
        clf2 = LogisticRegression(C=1.0, max_iter=500)
        clf2.fit(X_combo[tr_idx], y[tr_idx])
        preds_combo[val_idx] = clf2.predict_proba(X_combo[val_idx])[:, 1]
        
    acc_text = accuracy_score(y, (preds_text > 0.5).astype(int))
    acc_combo = accuracy_score(y, (preds_combo > 0.5).astype(int))
    print(f"🔬 {m1} vs {m2}:")
    print(f"   Text n-grams only:       {acc_text*100:.2f}%")
    print(f"   Text + Forensic Features: {acc_combo*100:.2f}% (Gain: {acc_combo*100 - acc_text*100:+.2f}%) 🚀")

test_pair_separation('gemini-3.7-flash', 'gemini-3.8-flash')
test_pair_separation('gpt-5.6-sol', 'gpt-5.6-terra')
test_pair_separation('glm-5.3', 'glm-5.3-flash')
test_pair_separation('claude-fable-5', 'glm-5.3-flash')

