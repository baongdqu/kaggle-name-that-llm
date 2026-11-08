import numpy as np, pandas as pd, re, time
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, diags
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

t0 = time.time()
print("🚀 Loading data & precomputed base logits...", flush=True)

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

# Define anchor models vs sister models
ANCHORS = ['kimi-k3', 'qwen3.8-max', 'claude-sonnet-5', 'gpt-6-astra', 'gemini-3.1-pro-preview', 'gpt-5.5']
anchor_idx = [L2I[m] for m in ANCHORS]

SISTERS = {
    'opus': ['claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5'],
    'gemini_flash': ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'],
    'gpt_56': ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra'],
    'glm': ['glm-5.3', 'glm-5.3-flash', 'claude-fable-5']
}

print(f"Total anchor models: {len(ANCHORS)}, Total sister models: {sum(len(v) for v in SISTERS.values())}")

# Baseline Global Hungarian Matching
def standard_hungarian(logits, temp=0.35):
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

# Two-Stage Anchor-Locked Matching
def anchor_locked_matching(logits, temp=0.35, margin_thresh=0.40):
    preds = np.zeros(len(train), dtype=int)
    locked_count = 0
    
    for p in unique_pids:
        idx = np.where(all_pids == p)[0]
        M = logits[idx].copy()
        
        # Softmax probabilities per row
        exp_M = np.exp(M - M.max(axis=1, keepdims=True))
        probs = exp_M / exp_M.sum(axis=1, keepdims=True)
        
        assigned_rows = set()
        assigned_cols = set()
        
        # Stage 1: Lock high-confidence anchor predictions
        for r in range(len(idx)):
            top_classes = np.argsort(probs[r])[::-1]
            c1, c2 = top_classes[0], top_classes[1]
            margin = probs[r, c1] - probs[r, c2]
            
            if c1 in anchor_idx and margin >= margin_thresh:
                if c1 not in assigned_cols:
                    preds[idx[r]] = c1
                    assigned_rows.add(r)
                    assigned_cols.add(c1)
                    locked_count += 1
                    
        # Stage 2: Solve Hungarian for remaining rows & columns
        rem_rows = [r for r in range(len(idx)) if r not in assigned_rows]
        rem_cols = [c for c in range(18) if c not in assigned_cols]
        
        if len(rem_rows) > 0 and len(rem_cols) > 0:
            M_sub = M[np.ix_(rem_rows, rem_cols)]
            K_sub = np.exp(M_sub / temp)
            for _ in range(15):
                K_sub = K_sub / (K_sub.sum(axis=1, keepdims=True) + 1e-12)
                K_sub = K_sub / (K_sub.sum(axis=0, keepdims=True) + 1e-12)
            r_sub, c_sub = linear_sum_assignment(-K_sub)
            for ri, ci in zip(r_sub, c_sub):
                preds[idx[rem_rows[ri]]] = rem_cols[ci]
                
    return balanced_accuracy_score(y_true, preds), locked_count

acc_std = standard_hungarian(log_dl, temp=0.35)
print(f"👉 Baseline Standard Hungarian (18x18 Global): {acc_std*100:.2f}%")

for m_th in [0.20, 0.30, 0.40, 0.50, 0.60]:
    acc_lock, n_lock = anchor_locked_matching(log_dl, temp=0.35, margin_thresh=m_th)
    print(f"🔒 Anchor-Locked (margin >= {m_th:.2f}): {acc_lock*100:.2f}% (Locked {n_lock}/{len(train)} samples, Gain: {acc_lock*100 - acc_std*100:+.2f}%)")
