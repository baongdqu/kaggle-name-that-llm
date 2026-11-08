import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

train = pd.read_csv('data/train.csv')
LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
L2I = {l: i for i, l in enumerate(LABELS)}
y_true = np.array([L2I[m] for m in train['model'].values])

all_pids = train['prompt_id'].values
unique_pids = np.unique(all_pids)
texts = train['text'].astype(str).tolist()

# Let's isolate GLM, Qwen, Kimi:
# 'glm-5.3', 'glm-5.3-flash', 'qwen3.8-max', 'kimi-k3'
target_models = ['glm-5.3', 'glm-5.3-flash', 'qwen3.8-max', 'kimi-k3']
target_idx = [L2I[m] for m in target_models]
mask = np.isin(y_true, target_idx)
sub_idx = np.where(mask)[0]

y_sub = np.array([target_models.index(LABELS[y_true[i]]) for i in sub_idx])
sub_texts = [texts[i] for i in sub_idx]
sub_pids = all_pids[sub_idx]

v = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=30000, sublinear_tf=True)
X = v.fit_transform(sub_texts)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(sub_idx))

for tr_p, va_p in kf.split(np.unique(sub_pids)):
    tr_pids = np.unique(sub_pids)[tr_p]
    va_pids = np.unique(sub_pids)[va_p]
    tr_i = np.where(np.isin(sub_pids, tr_pids))[0]
    va_i = np.where(np.isin(sub_pids, va_pids))[0]
    
    clf = LinearSVC(C=0.5, random_state=42)
    clf.fit(X[tr_i], y_sub[tr_i])
    oof[va_i] = clf.predict(X[va_i])

acc = balanced_accuracy_score(y_sub, oof)
print(f"🇨🇳 Classical ML Balanced Accuracy on Chinese LLM Cluster (GLM vs Qwen vs Kimi): {acc*100:.2f}% (Random is 25.0%)")
for m_idx, m_name in enumerate(target_models):
    m_mask = (y_sub == m_idx)
    sub_acc = (oof[m_mask] == m_idx).mean()
    print(f"   - {m_name:15s}: {sub_acc*100:.2f}%")
