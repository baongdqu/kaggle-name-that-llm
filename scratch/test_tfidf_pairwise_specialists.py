import numpy as np, pandas as pd, re, time
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score

train = pd.read_csv('data/train.csv')
texts = train['text'].astype(str).tolist()
all_pids = train['prompt_id'].values

LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
L2I = {l: i for i, l in enumerate(LABELS)}
y_true = np.array([L2I[m] for m in train['model'].values])

print("📚 Building High-Resolution Pairwise TF-IDF...", flush=True)
v_char = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', min_df=2, max_features=40000, sublinear_tf=True, dtype=np.float32)
v_word = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', min_df=2, max_features=25000, sublinear_tf=True, dtype=np.float32)

X_tfidf = hstack([v_char.fit_transform(texts), v_word.fit_transform(texts)], format='csr')
print(f"TF-IDF Shape: {X_tfidf.shape}", flush=True)

gfk = GroupKFold(n_splits=5)

pairs = [
    ('gemini-3.7-flash', 'gemini-3.8-flash'),
    ('gpt-5.6-sol', 'gpt-5.6-terra'),
    ('gpt-5.6-luna', 'gpt-5.6-terra'),
    ('glm-5.3', 'glm-5.3-flash'),
    ('claude-fable-5', 'glm-5.3')
]

for m1, m2 in pairs:
    idx1, idx2 = L2I[m1], L2I[m2]
    sub_idx = [i for i in range(len(train)) if y_true[i] in [idx1, idx2]]
    y_sub = (y_true[sub_idx] == idx2).astype(int)
    X_sub = X_tfidf[sub_idx]
    pids_sub = all_pids[sub_idx]
    
    oof_preds = np.zeros(len(sub_idx))
    for tr, va in gfk.split(X_sub, y_sub, groups=pids_sub):
        clf = LinearSVC(C=0.5, random_state=42, max_iter=2000, dual=True)
        clf.fit(X_sub[tr], y_sub[tr])
        oof_preds[va] = clf.decision_function(X_sub[va])
        
    bacc = balanced_accuracy_score(y_sub, (oof_preds > 0).astype(int))
    print(f"👉 Pair '{m1}' vs '{m2}': TF-IDF LinearSVC OOF Balanced Accuracy = {bacc*100:.2f}%", flush=True)

