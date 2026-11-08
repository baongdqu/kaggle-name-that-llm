import numpy as np, pandas as pd, re, time
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
import sys
sys.path.insert(0, 'scratch')
import gauss as g

t0 = time.time()
print("🚀 Loading data...", flush=True)

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

m1 = d1['probs'][:, idx_map1]
m2 = d2['probs'][:, idx_map2]
m3 = d3['probs'][:, idx_map3]

dl_blend = (1.0 * m1 + 1.0 * m2 + 1.5 * m3) / 3.5
log_dl = np.log(dl_blend + 1e-12)

all_pids = train['prompt_id'].values
unique_pids = np.unique(all_pids)
texts = train['text'].astype(str).tolist()

print("📚 Extracting stylometric features...", flush=True)
v_char = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=15000, sublinear_tf=True, dtype=np.float32)
v_word = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', max_features=10000, sublinear_tf=True, dtype=np.float32)
X_sparse = hstack([v_char.fit_transform(texts), v_word.fit_transform(texts)], format='csr')

# Map prompt_id to integer groups 0..G-1
p2g = {p: i for i, p in enumerate(unique_pids)}
group_ids = np.array([p2g[p] for p in all_pids])

print(f"🎯 Fitting 5-Fold Shared-Prompt Gaussian Head...", flush=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_gauss = np.zeros((len(train), 18), dtype=np.float32)

for fold, (tr_pids_idx, val_pids_idx) in enumerate(kf.split(unique_pids)):
    t_f = time.time()
    tr_pids = unique_pids[tr_pids_idx]
    val_pids = unique_pids[val_pids_idx]
    tr_idx = np.where(np.isin(all_pids, tr_pids))[0]
    val_idx = np.where(np.isin(all_pids, val_pids))[0]
    
    # Dense reduced representation for Gaussian head
    # Use PCA/economy SVD on train fold
    prm = g.fit(X_sparse[tr_idx].toarray(), y_true[tr_idx], group_ids[tr_idx], cfg={"d": 128, "iters": 3})
    
    # Val group ids remapped
    val_g2idx = {g: i for i, g in enumerate(np.unique(group_ids[val_idx]))}
    val_local_groups = np.array([val_g2idx[g] for g in group_ids[val_idx]])
    
    sc = g.scores(X_sparse[val_idx].toarray(), val_local_groups, prm)
    oof_gauss[val_idx] = sc
    print(f"  Fold {fold+1}/5 done in {time.time()-t_f:.1f}s", flush=True)

def eval_matching(logits, temp=0.5):
    preds = np.zeros(len(train), dtype=int)
    for p in unique_pids:
        idx = np.where(all_pids == p)[0]
        M = logits[idx]
        K = np.exp(M / temp)
        for _ in range(15):
            K = K / (K.sum(axis=1, keepdims=True) + 1e-12)
            K = K / (K.sum(axis=0, keepdims=True) + 1e-12)
        r, c = linear_sum_assignment(-K)
        preds[idx[r]] = c
    return balanced_accuracy_score(y_true, preds)

acc_gauss_alone = eval_matching(oof_gauss)
print(f"\n👉 Gaussian Head Alone: {acc_gauss_alone*100:.2f}%")

for w_g in [0.2, 0.5, 0.7, 1.0, 1.5]:
    tot = log_dl + w_g * oof_gauss
    print(f"👉 DL + Gaussian Head (w={w_g}): {eval_matching(tot)*100:.2f}%")
