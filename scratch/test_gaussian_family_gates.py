import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.svm import LinearSVC
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
import re, time

t0 = time.time()
print("🚀 [Step 1] Loading Data and Deep Learning OOFs...", flush=True)

train = pd.read_csv('data/train.csv')
p1 = r'scratch\out_tokattn1\tokattn_oof.npz'
p2 = r'scratch\out_tokattn2\tokattn_q8l22_oof.npz'
p3 = r'scratch\out_tokattn2\tokattn_debl23_oof.npz'

d1 = np.load(p1, allow_pickle=True)
d2 = np.load(p2, allow_pickle=True)
d3 = np.load(p3, allow_pickle=True)

LABELS = ['claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
 'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash','glm-5.3','glm-5.3-flash',
 'gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra','kimi-k3','qwen3.8-max']
L2I = {l: i for i, l in enumerate(LABELS)}
y_true = np.array([L2I[m] for m in train['model'].values])

idx_map1 = [list(d1['classes']).index(l) for l in LABELS]
idx_map2 = [list(d2['classes']).index(l) for l in LABELS]
idx_map3 = [list(d3['classes']).index(l) for l in LABELS]

m1 = d1['probs'][:, idx_map1]
m2 = d2['probs'][:, idx_map2]
m3 = d3['probs'][:, idx_map3]

# Optimal DL weights from our earlier grid search: (1.0, 1.0, 1.5)
dl_blend = (1.0 * m1 + 1.0 * m2 + 1.5 * m3) / 3.5
log_dl = np.log(dl_blend + 1e-12)

all_pids = train['prompt_id'].values
unique_pids = np.unique(all_pids)

# Check DL baseline
preds_dl = np.zeros(len(train), dtype=int)
for p in unique_pids:
    idx = np.where(all_pids == p)[0]
    r, c = linear_sum_assignment(-log_dl[idx])
    preds_dl[idx[r]] = c
print(f"✅ DL Ensemble (Qwen14B + Q8B + DeBERTa) + Hungarian: {balanced_accuracy_score(y_true, preds_dl)*100:.2f}%", flush=True)

print("\n🛠️ [Step 2] Engineering Multi-View Features...", flush=True)
texts = train['text'].astype(str).tolist()
_FW = set('a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little'.split())
_shape = lambda t: re.sub(r'[^\W\d_]+', lambda m_: 'X' if m_[0].isupper() else 'x', t)
_func = lambda t: ' '.join(w.lower() if w.lower() in _FW or not w.isalnum() else 'CONTENT' for w in re.findall(r'\w+|[^\w\s]', t))

matrices = []
for view, lo, hi, mf, wv in [
    ('char', 3, 5, 30000, 1.0),
    ('word', 1, 3, 20000, 0.7),
    ('shape', 2, 6, 10000, 0.5),
    ('function', 1, 3, 10000, 0.5)
]:
    prep = _shape if view == 'shape' else _func if view == 'function' else (lambda s_: s_)
    an = 'char' if view == 'shape' else 'word' if view == 'function' else view
    o = dict(analyzer=an, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf, dtype=np.float32)
    if an == 'word': o['token_pattern'] = r'(?u)\b\w+\b|[^\w\s]'
    v = TfidfVectorizer(**o)
    matrices.append(v.fit_transform([prep(t) for t in texts]) * wv)

X_all = hstack(matrices, format='csr')

# Create Prompt-Centered Feature Matrix X_centered
# x_centered = x - mean_prompt(x)
print("Centering sparse feature matrix per prompt...", flush=True)
X_centered = X_all.copy().tolil()
for p in unique_pids:
    idx = np.where(all_pids == p)[0]
    sub_mean = np.array(X_all[idx].mean(axis=0))
    X_centered[idx] = X_all[idx] - sub_mean
X_centered = X_centered.tocsr()

print(f"Features ready in {time.time()-t0:.1f}s. Running 5-Fold GroupKFold CV...", flush=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_svc_plain = np.zeros((len(train), 18), dtype=np.float32)
oof_svc_centered = np.zeros((len(train), 18), dtype=np.float32)

# Define Family Gates
FAMILIES = {
    'gemini_flash': ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'],
    'gpt56': ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra'],
    'glm_fable': ['glm-5.3', 'glm-5.3-flash', 'claude-fable-5'],
    'opus': ['claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5']
}

oof_gates = {f: np.zeros((len(train), len(m)), dtype=np.float32) for f, m in FAMILIES.items()}

for fold, (tr_pids_idx, val_pids_idx) in enumerate(kf.split(unique_pids)):
    tr_pids = unique_pids[tr_pids_idx]
    val_pids = unique_pids[val_pids_idx]
    tr_idx = np.where(np.isin(all_pids, tr_pids))[0]
    val_idx = np.where(np.isin(all_pids, val_pids))[0]
    
    # 1. Plain Global SVC
    svc_p = LinearSVC(C=0.5, random_state=42, max_iter=2000, dual=True)
    svc_p.fit(X_all[tr_idx], y_true[tr_idx])
    dp = svc_p.decision_function(X_all[val_idx])
    exp_dp = np.exp(dp - dp.max(axis=1, keepdims=True))
    oof_svc_plain[val_idx] = np.log(exp_dp / exp_dp.sum(axis=1, keepdims=True) + 1e-12)
    
    # 2. Centered Global SVC (Prompt-Invariant Author Signal)
    svc_c = LinearSVC(C=0.5, random_state=42, max_iter=2000, dual=True)
    svc_c.fit(X_centered[tr_idx], y_true[tr_idx])
    dc = svc_c.decision_function(X_centered[val_idx])
    exp_dc = np.exp(dc - dc.max(axis=1, keepdims=True))
    oof_svc_centered[val_idx] = np.log(exp_dc / exp_dc.sum(axis=1, keepdims=True) + 1e-12)
    
    # 3. Family Specialized Gates (Trained on centered features)
    for fname, fmodels in FAMILIES.items():
        f_indices = [L2I[m] for m in fmodels]
        sub_tr = [i for i in tr_idx if y_true[i] in f_indices]
        sub_y = [f_indices.index(y_true[i]) for i in sub_tr]
        
        gate_svc = LinearSVC(C=1.0, random_state=42, max_iter=2000, dual=True)
        gate_svc.fit(X_centered[sub_tr], sub_y)
        d_gate = gate_svc.decision_function(X_centered[val_idx])
        exp_g = np.exp(d_gate - d_gate.max(axis=1, keepdims=True))
        oof_gates[fname][val_idx] = np.log(exp_g / exp_g.sum(axis=1, keepdims=True) + 1e-12)
        
    print(f"  Fold {fold+1}/5 completed.", flush=True)

# Grid Search over weights
print("\n" + "="*50)
print("Evaluating Blends and Family Gates...")

def evaluate_blend(w_plain, w_cent, w_gate, temp=0.5):
    preds = np.zeros(len(train), dtype=int)
    tot_log = log_dl + w_plain * oof_svc_plain + w_cent * oof_svc_centered
    
    if w_gate > 0:
        tot_log = tot_log.copy()
        for fname, fmodels in FAMILIES.items():
            f_indices = [L2I[m] for m in fmodels]
            # Family mass preservation: reweight inside family according to gate logits
            gate_logits = oof_gates[fname]
            for k, glob_cls in enumerate(f_indices):
                tot_log[:, glob_cls] += w_gate * gate_logits[:, k]
                
    for p in unique_pids:
        idx = np.where(all_pids == p)[0]
        M = tot_log[idx]
        
        # Soft-Sinkhorn balancing
        K = np.exp(M / temp)
        for _ in range(15):
            K = K / (K.sum(axis=1, keepdims=True) + 1e-12)
            K = K / (K.sum(axis=0, keepdims=True) + 1e-12)
            
        r, c = linear_sum_assignment(-K)
        preds[idx[r]] = c
        
    return balanced_accuracy_score(y_true, preds)

print(f"1. Plain SVC only (w=2.0):                {evaluate_blend(2.0, 0.0, 0.0)*100:.2f}%")
print(f"2. Centered SVC only (w=2.0):             {evaluate_blend(0.0, 2.0, 0.0)*100:.2f}%")
print(f"3. Dual-SVC (Plain 1.5 + Centered 1.5):   {evaluate_blend(1.5, 1.5, 0.0)*100:.2f}%")
print(f"4. Dual-SVC + Family Gate (w_gate=1.0):   {evaluate_blend(1.5, 1.5, 1.0)*100:.2f}%")
print(f"5. Dual-SVC + Family Gate (w_gate=2.0):   {evaluate_blend(1.5, 1.5, 2.0)*100:.2f}%")
print(f"6. Dual-SVC + Family Gate (w_gate=3.0):   {evaluate_blend(1.0, 2.0, 2.0)*100:.2f}%")

best_score = 0
best_params = None
for wp in [1.0, 1.5, 2.0]:
    for wc in [1.0, 1.5, 2.0]:
        for wg in [0.5, 1.0, 1.5, 2.0, 3.0]:
            for tmp in [0.35, 0.5, 0.7]:
                s = evaluate_blend(wp, wc, wg, temp=tmp)
                if s > best_score:
                    best_score = s
                    best_params = (wp, wc, wg, tmp)
                    print(f"🔥 BEST: {best_score*100:.2f}% | wp={wp}, wc={wc}, wg={wg}, temp={tmp}")

print("\n" + "="*50)
print(f"🏆 Optimal SOTA Score: {best_score*100:.2f}% with params {best_params}")

