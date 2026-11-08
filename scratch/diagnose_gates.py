import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, diags
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
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

# 1. Check Gemini Flash 3-way accuracy
gemini_models = ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash']
gem_sub = train[train['model'].isin(gemini_models)].copy()
print("Gemini counts:", gem_sub['model'].value_counts())

# Test character n-grams + word n-grams on gemini subset
v_c = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=25000, sublinear_tf=True)
X_gem = v_c.fit_transform(gem_sub['text'].values)
y_gem = np.array([gemini_models.index(m) for m in gem_sub['model'].values])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds_gem = np.zeros(len(gem_sub), dtype=int)
gem_pids = gem_sub['prompt_id'].values
gem_upids = np.unique(gem_pids)

for tr_p_idx, val_p_idx in kf.split(gem_upids):
    tr_p = gem_upids[tr_p_idx]
    val_p = gem_upids[val_p_idx]
    tr_idx = np.where(np.isin(gem_pids, tr_p))[0]
    val_idx = np.where(np.isin(gem_pids, val_p))[0]
    
    svc = LinearSVC(C=0.2, tol=1e-2, random_state=42, dual=True)
    svc.fit(X_gem[tr_idx], y_gem[tr_idx])
    preds_gem[val_idx] = svc.predict(X_gem[val_idx])

print(f"👉 Gemini Flash 3-Way CV Accuracy: {accuracy_score(y_gem, preds_gem)*100:.2f}% (random baseline 33.33%)")

# 2. Check GPT-5.6 3-way accuracy
gpt_models = ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra']
gpt_sub = train[train['model'].isin(gpt_models)].copy()
v_gpt = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=25000, sublinear_tf=True)
X_gpt = v_gpt.fit_transform(gpt_sub['text'].values)
y_gpt = np.array([gpt_models.index(m) for m in gpt_sub['model'].values])

preds_gpt = np.zeros(len(gpt_sub), dtype=int)
gpt_pids = gpt_sub['prompt_id'].values
gpt_upids = np.unique(gpt_pids)

for tr_p_idx, val_p_idx in kf.split(gpt_upids):
    tr_p = gpt_upids[tr_p_idx]
    val_p = gpt_upids[val_p_idx]
    tr_idx = np.where(np.isin(gpt_pids, tr_p))[0]
    val_idx = np.where(np.isin(gpt_pids, val_p))[0]
    
    svc = LinearSVC(C=0.2, tol=1e-2, random_state=42, dual=True)
    svc.fit(X_gpt[tr_idx], y_gpt[tr_idx])
    preds_gpt[val_idx] = svc.predict(X_gpt[val_idx])

print(f"👉 GPT-5.6 3-Way CV Accuracy: {accuracy_score(y_gpt, preds_gpt)*100:.2f}% (random baseline 33.33%)")

