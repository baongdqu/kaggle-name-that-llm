import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

train = pd.read_csv('data/train.csv')
p1 = r'scratch\out_tokattn1\tokattn_oof.npz'
p2 = r'scratch\out_tokattn2\tokattn_q8l22_oof.npz'
p3 = r'scratch\out_tokattn2\tokattn_debl23_oof.npz'

d1 = np.load(p1, allow_pickle=True)
d2 = np.load(p2, allow_pickle=True)
d3 = np.load(p3, allow_pickle=True)

LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
L2I = {l: i for i, l in enumerate(LABELS)}
y_true = np.array([L2I[m] for m in train['model'].values])

idx_map1 = [list(d1['classes']).index(l) for l in LABELS]
idx_map2 = [list(d2['classes']).index(l) for l in LABELS]
idx_map3 = [list(d3['classes']).index(l) for l in LABELS]

dl_blend = (1.0 * d1['probs'][:, idx_map1] + 1.0 * d2['probs'][:, idx_map2] + 1.5 * d3['probs'][:, idx_map3]) / 3.5

all_pids = train['prompt_id'].values
unique_pids = np.unique(all_pids)
texts = train['text'].astype(str).tolist()

target_models = ['glm-5.3', 'glm-5.3-flash', 'qwen3.8-max', 'kimi-k3']
target_idx = [L2I[m] for m in target_models]
mask = np.isin(y_true, target_idx)
sub_idx = np.where(mask)[0]

y_sub = np.array([target_models.index(LABELS[y_true[i]]) for i in sub_idx])
sub_texts = [texts[i] for i in sub_idx]
sub_pids = all_pids[sub_idx]
dl_sub = dl_blend[sub_idx][:, target_idx]
dl_sub_norm = dl_sub / dl_sub.sum(axis=1, keepdims=True)

v = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=30000, sublinear_tf=True)
X = v.fit_transform(sub_texts)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_ml_logits = np.zeros((len(sub_idx), 4))

for tr_p, va_p in kf.split(np.unique(sub_pids)):
    tr_pids = np.unique(sub_pids)[tr_p]
    va_pids = np.unique(sub_pids)[va_p]
    tr_i = np.where(np.isin(sub_pids, tr_pids))[0]
    va_i = np.where(np.isin(sub_pids, va_pids))[0]
    
    clf = LinearSVC(C=0.5, random_state=42)
    clf.fit(X[tr_i], y_sub[tr_i])
    oof_ml_logits[va_i] = clf.decision_function(X[va_i])

exp_l = np.exp(oof_ml_logits - oof_ml_logits.max(axis=1, keepdims=True))
ml_probs = exp_l / exp_l.sum(axis=1, keepdims=True)

acc_dl = balanced_accuracy_score(y_sub, np.argmax(dl_sub_norm, axis=1))
acc_ml = balanced_accuracy_score(y_sub, np.argmax(ml_probs, axis=1))

for w in [0.0, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0]:
    blend = dl_sub_norm + w * ml_probs
    acc_b = balanced_accuracy_score(y_sub, np.argmax(blend, axis=1))
    print(f"Weight ML {w:.1f}: Balanced Acc = {acc_b*100:.2f}% (DL={acc_dl*100:.2f}%, ML={acc_ml*100:.2f}%)")
