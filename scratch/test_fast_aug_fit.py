import numpy as np, pandas as pd, time
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, vstack
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier

t0 = time.time()
train = pd.read_csv('data/train.csv')
df_ext = pd.read_parquet('data/external_short_wide.parquet')

LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
L2I = {l: i for i, l in enumerate(LABELS)}

ext_texts, ext_labels = [], []
for col in LABELS:
    if col in df_ext.columns:
        s = df_ext[col].dropna()
        s_sampled = s.sample(min(len(s), 500), random_state=42)
        for t in s_sampled:
            ext_texts.append(str(t))
            ext_labels.append(L2I[col])

print(f"Extracted {len(ext_texts)} external samples in {time.time()-t0:.2f}s")

all_texts = train['text'].astype(str).tolist() + ext_texts
y_all = np.concatenate([np.array([L2I[m] for m in train['model'].values]), np.array(ext_labels)])

t1 = time.time()
v1 = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=35000, sublinear_tf=True)
v2 = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', max_features=25000, sublinear_tf=True)
X = hstack([v1.fit_transform(all_texts), v2.fit_transform(all_texts)], format='csr')
print(f"TFIDF ({X.shape}) created in {time.time()-t1:.2f}s")

t2 = time.time()
clf = RidgeClassifier(alpha=1.0, random_state=42)
clf.fit(X, y_all)
print(f"RidgeClassifier fit on {X.shape[0]} samples took: {time.time()-t2:.2f}s")

t3 = time.time()
svc = LinearSVC(C=0.3, tol=1e-3, random_state=42, max_iter=1000)
svc.fit(X, y_all)
print(f"LinearSVC (tol=1e-3) fit on {X.shape[0]} samples took: {time.time()-t3:.2f}s")

