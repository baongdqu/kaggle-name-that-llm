import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
import re, time

t0 = time.time()
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

dl_blend = (d1['probs'][:, idx_map1] + d2['probs'][:, idx_map2] + d3['probs'][:, idx_map3]) / 3.0
log_dl = np.log(dl_blend + 1e-12)

# Multi-view TFIDF
texts = train['text'].astype(str).tolist()
_FW = set('a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little'.split())
_shape = lambda t: re.sub(r'[^\W\d_]+', lambda m_: 'X' if m_[0].isupper() else 'x', t)
_func = lambda t: ' '.join(w.lower() if w.lower() in _FW or not w.isalnum() else 'CONTENT' for w in re.findall(r'\w+|[^\w\s]', t))

matrices = []
for view, lo, hi, mf, wv in [
    ('char', 3, 5, 35000, 1.0),
    ('char_wb', 2, 4, 25000, 0.8),
    ('word', 1, 3, 20000, 0.7),
    ('shape', 2, 6, 12000, 0.5),
    ('function', 1, 3, 8000, 0.5)
]:
    prep = _shape if view == 'shape' else _func if view == 'function' else (lambda s_: s_)
    an = 'char' if view == 'shape' else 'word' if view == 'function' else ('char_wb' if view == 'char_wb' else view)
    o = dict(analyzer=an, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf, dtype=np.float32)
    if an == 'word': o['token_pattern'] = r'(?u)\b\w+\b|[^\w\s]'
    v = TfidfVectorizer(**o)
    matrices.append(v.fit_transform([prep(t) for t in texts]) * wv)

X_all = hstack(matrices, format='csr')
print(f"Features: {X_all.shape} in {time.time()-t0:.1f}s")

# Extract stylometric metrics per sample
wlens = np.array([len(t.split()) for t in texts], dtype=np.float32)
em_dashes = np.array([t.count('—') for t in texts], dtype=np.float32)
colons = np.array([t.count(':') for t in texts], dtype=np.float32)
semis = np.array([t.count(';') for t in texts], dtype=np.float32)
curly = np.array([t.count('’') + t.count('“') for t in texts], dtype=np.float32)

all_pids = train['prompt_id'].values
unique_pids = np.unique(all_pids)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Precompute LinearSVC OOF
oof_svc = np.zeros((len(train), 18), dtype=np.float32)
for fold, (tr_pids_idx, val_pids_idx) in enumerate(kf.split(unique_pids)):
    tr_pids = unique_pids[tr_pids_idx]
    val_pids = unique_pids[val_pids_idx]
    tr_idx = np.where(np.isin(all_pids, tr_pids))[0]
    val_idx = np.where(np.isin(all_pids, val_pids))[0]
    
    svc = LinearSVC(C=0.5, random_state=42, max_iter=2000, dual=True)
    svc.fit(X_all[tr_idx], y_true[tr_idx])
    d_val = svc.decision_function(X_all[val_idx])
    d_m = d_val.max(axis=1, keepdims=True)
    exp_d = np.exp(d_val - d_m)
    oof_svc[val_idx] = np.log(exp_d / exp_d.sum(axis=1, keepdims=True) + 1e-12)

print(f"LinearSVC 5-fold OOF precomputed!")

# Baseline with w_svc = 2.0
base_log = log_dl + 2.0 * oof_svc
preds_base = np.zeros(len(train), dtype=int)
for pid in unique_pids:
    idx = np.where(all_pids == pid)[0]
    cost = -base_log[idx]
    r, c = linear_sum_assignment(cost)
    preds_base[idx[r]] = c
acc_base = balanced_accuracy_score(y_true, preds_base)
print(f"👉 Baseline (DL + 2.0 * LinearSVC): {acc_base*100:.2f}%")

# Now let's test Prompt-Relative Stylometric Bonus Matrix
# For each prompt, we adjust the cost matrix using intra-prompt rankings!
def evaluate_intra_prompt(bonus_g37_len, bonus_opus_semi, bonus_opus_colon, bonus_sol_curly):
    preds = np.zeros(len(train), dtype=int)
    g37_idx = L2I['gemini-3.7-flash']
    g38_idx = L2I['gemini-3.8-flash']
    op5_idx = L2I['claude-opus-5']
    op48_idx = L2I['claude-opus-4-8']
    op46_idx = L2I['claude-opus-4-6']
    sol_idx = L2I['gpt-5.6-sol']
    terra_idx = L2I['gpt-5.6-terra']
    
    for pid in unique_pids:
        idx = np.where(all_pids == pid)[0]
        cost = -base_log[idx].copy() # shape (18, 18)
        
        # Intra-prompt length ranking for Gemini Flash
        sub_w = wlens[idx]
        rank_w = np.argsort(np.argsort(sub_w)) # 0 to 17
        norm_w = (rank_w - 8.5) / 8.5 # -1.0 to +1.0
        
        # Intra-prompt punctuation ranking
        sub_semi = semis[idx]
        norm_semi = (sub_semi - sub_semi.mean()) / max(0.5, sub_semi.std())
        
        sub_colon = colons[idx]
        norm_colon = (sub_colon - sub_colon.mean()) / max(0.5, sub_colon.std())
        
        sub_curly = curly[idx]
        norm_curly = (sub_curly - sub_curly.mean()) / max(0.5, sub_curly.std())
        
        # Adjust costs (-cost means higher probability)
        # G3.7 is longer than G3.8
        cost[:, g37_idx] -= bonus_g37_len * norm_w
        cost[:, g38_idx] += bonus_g37_len * norm_w
        
        # Opus-5 has more semicolons and colons than Opus-4-8
        cost[:, op5_idx] -= (bonus_opus_semi * norm_semi + bonus_opus_colon * norm_colon)
        cost[:, op48_idx] += (bonus_opus_semi * norm_semi + bonus_opus_colon * norm_colon)
        
        # Sol has more curly quotes than Terra
        cost[:, sol_idx] -= bonus_sol_curly * norm_curly
        cost[:, terra_idx] += bonus_sol_curly * norm_curly
        
        r, c = linear_sum_assignment(cost)
        preds[idx[r]] = c
        
    return balanced_accuracy_score(y_true, preds)

print("\n🔍 Testing Intra-Prompt Relative Stylometric Bonus...")
for bg in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    for b_semi in [0.0, 0.2, 0.4, 0.6]:
        acc = evaluate_intra_prompt(bonus_g37_len=bg, bonus_opus_semi=b_semi, bonus_opus_colon=0.2, bonus_sol_curly=0.2)
        if acc > acc_base:
            print(f"🔥 GAIN: {acc*100:.2f}% (+{(acc-acc_base)*100:.2f}%) | bg={bg:.1f}, b_semi={b_semi:.1f}")

