import numpy as np, pandas as pd, re, time
from scipy.special import logsumexp, softmax
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from functools import lru_cache

t0 = time.time()
print("🚀 [Step 1] Loading Datasets and Deep Learning Ensemble...", flush=True)

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
all_pids = train['prompt_id'].values
unique_pids = np.unique(all_pids)
texts = train['text'].astype(str).tolist()

print("📚 [Step 2] Building Stylometric Forensic Representations...", flush=True)

MARKERS = [',','.', ';', ':', '!', '?','—','–','-','’','‘','“','”','"',"'",'(',')','[',']','{','}','*','`','•','…','/','\\',':','\n','\n\n','\n-','\n1.','\n2.','##','```','**','> ','\t',' = ', ' e.g.', ' i.e.', 'However,','Overall,','In conclusion','Therefore,','Moreover,','First,','Finally,',' - ',': ', ' ; ', '  ']
WORDS = ['the','and','but','that','which','with','into','from','for','you','your','our','we','I','it','its','they','their','there','this','these','those','however','therefore','moreover','although','because','perhaps','might','would','could','should','like','such','just','very','also','not','if','or','as','so','by','of','to','in','on','at','is','are','be','was','were','have','has','had','may','must','all','any','each','every','many','some','more','most','only','even','still','then','when','where','why','how','what','who']
WRE = re.compile(r"[A-Za-z]+(?:[’'][A-Za-z]+)?")
SRE = re.compile(r'(?<=[.?!])\s+')

@lru_cache(maxsize=15000)
def extract_row_features(t):
    t = str(t)
    s = WRE.findall(t); lo=[z.lower() for z in s]; n=max(len(s),1); nt=max(len(t),1); ns=max(len(t.splitlines()),1)
    lens=np.array([len(z) for z in s] or [0]); sent_lens=np.array([len(WRE.findall(z)) for z in SRE.split(t) if z.strip()] or [0])
    base = [np.log1p(n),np.log1p(nt), nt/n,ns/n,len(set(lo))/n, float(np.mean(lens)), float(np.std(lens)), float(np.mean(sent_lens)), float(np.std(sent_lens)),float(np.max(sent_lens)),float(np.mean(lens>=8)),sum(c.isupper() for c in t)/nt, sum(c.isdigit() for c in t)/nt,sum(ord(c)>127 for c in t)/nt, sum(c.isspace() for c in t)/nt, sum(c=='\n' for c in t)/nt]
    base += [min(t.count(m)/n, 3) for m in MARKERS]
    cnt = {}
    for w in lo: cnt[w]=cnt.get(w,0)+1
    base += [cnt.get(w,0)/n for w in WORDS]
    return np.array(base, dtype=np.float32)

X_feats = np.stack([extract_row_features(t) for t in texts])
print(f"✅ Extracted {X_feats.shape[1]} stylometric features in {time.time()-t0:.1f}s!", flush=True)

print("🔬 [Step 3] Loading Classical ML (Centered Dual-SVC)...", flush=True)
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, diags
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold

matrices = []
for view, lo, hi, mf, wv in [
    ('char', 3, 5, 35000, 1.0),
    ('char_wb', 2, 4, 25000, 0.8),
    ('word', 1, 3, 20000, 0.7),
]:
    v = TfidfVectorizer(ngram_range=(lo, hi), analyzer=view, lowercase=False, sublinear_tf=True, min_df=2, max_features=mf, dtype=np.float32)
    matrices.append(v.fit_transform(texts) * wv)

X_all = hstack(matrices, format='csr')

def sparse_group_centre(X, groups, lo=2, hi=18):
    X = csr_matrix(X, dtype=np.float32)
    _, inv, cnt = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    keep = np.flatnonzero((cnt >= lo) & (cnt <= hi))
    if not len(keep): return X.copy()
    col = np.full(len(cnt), -1); col[keep] = np.arange(len(keep))
    rows = np.flatnonzero(col[inv] >= 0)
    M = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, col[inv][rows])), shape=(X.shape[0], len(keep)), dtype=np.float32)
    means = diags(np.array(1.0 / cnt[keep], dtype=np.float32)) @ (M.T @ X)
    return (X - M @ means).tocsr()

X_centered = sparse_group_centre(X_all, all_pids)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_svc_plain = np.zeros((len(train), 18), dtype=np.float32)
oof_svc_centered = np.zeros((len(train), 18), dtype=np.float32)

for fold, (tr_pids_idx, val_pids_idx) in enumerate(kf.split(unique_pids)):
    tr_pids = unique_pids[tr_pids_idx]
    val_pids = unique_pids[val_pids_idx]
    tr_idx = np.where(np.isin(all_pids, tr_pids))[0]
    val_idx = np.where(np.isin(all_pids, val_pids))[0]
    
    svc_p = LinearSVC(C=0.5, random_state=42, max_iter=2000, dual=True)
    svc_p.fit(X_all[tr_idx], y_true[tr_idx])
    dp = svc_p.decision_function(X_all[val_idx])
    exp_dp = np.exp(dp - dp.max(axis=1, keepdims=True))
    oof_svc_plain[val_idx] = np.log(exp_dp / exp_dp.sum(axis=1, keepdims=True) + 1e-12)
    
    svc_c = LinearSVC(C=0.5, random_state=42, max_iter=2000, dual=True)
    svc_c.fit(X_centered[tr_idx], y_true[tr_idx])
    dc = svc_c.decision_function(X_centered[val_idx])
    exp_dc = np.exp(dc - dc.max(axis=1, keepdims=True))
    oof_svc_centered[val_idx] = np.log(exp_dc / exp_dc.sum(axis=1, keepdims=True) + 1e-12)

log_dl = np.log(dl_blend + 1e-12)
base_logits = log_dl + 1.5 * oof_svc_plain + 1.5 * oof_svc_centered

# Also add the Opus external gate
claude_ext = pd.read_csv('data/ntl_extra_passages_claude.csv')
opus_models = ['claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5']
sub_ext_opus = claude_ext[claude_ext['model'].isin(opus_models)].copy().reset_index(drop=True)
y_opus_train = np.array([opus_models.index(m) for m in sub_ext_opus['model'].values])

v_opus1 = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=35000, sublinear_tf=True, dtype=np.float32)
v_opus2 = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', max_features=25000, sublinear_tf=True, dtype=np.float32)
X_op_tr = hstack([v_opus1.fit_transform(sub_ext_opus['text'].values), v_opus2.fit_transform(sub_ext_opus['text'].values)], format='csr')
X_op_va = hstack([v_opus1.transform(texts), v_opus2.transform(texts)], format='csr')

clf_opus = LinearSVC(C=0.5, random_state=42, max_iter=2000, dual=True)
clf_opus.fit(X_op_tr, y_opus_train)
logits_opus = clf_opus.decision_function(X_op_va)
opus_idx = [L2I[m] for m in opus_models]

sc_opus = logits_opus - logits_opus.mean(axis=1, keepdims=True)
sc_opus = sc_opus / np.maximum(sc_opus.std(axis=1, keepdims=True), 0.25)
base_logits[:, opus_idx] += 0.3 * sc_opus

print("⚖️ [Step 4] Applying Conserving Sibling-Style Residuals...", flush=True)

def eval_matching(logits, temp=0.35):
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

acc_init = eval_matching(base_logits)
print(f"👉 Baseline Score (Dual-SVC + Opus Gate): {acc_init*100:.2f}%", flush=True)

# Function to apply OOF probability-conserving sibling style residual
def apply_conserving_residual(logits_in, max_alpha=0.20, margin_thresh=0.65):
    out = np.asarray(logits_in, dtype=np.float64).copy()
    gfk = GroupKFold(n_splits=5)
    
    # Define family clusters to refine
    CLUSTERS = [
        ('gemini_flash', [L2I['gemini-3.6-flash'], L2I['gemini-3.7-flash'], L2I['gemini-3.8-flash']], 0.01),
        ('gpt56', [L2I['gpt-5.6-luna'], L2I['gpt-5.6-sol'], L2I['gpt-5.6-terra']], 0.01),
        ('glm', [L2I['glm-5.3'], L2I['glm-5.3-flash']], 0.05),
    ]
    
    for fname, cols, C in CLUSTERS:
        cols = np.array(cols)
        train_mask = np.isin(y_true, cols)
        inds = np.flatnonzero(train_mask)
        X = X_feats[inds]
        target = np.searchsorted(cols, y_true[inds])
        grp = all_pids[inds]
        
        oof_probs = np.zeros((len(train), len(cols)), dtype=np.float32)
        oof_preds = np.empty(len(target), dtype=np.int32)
        
        for a, b in gfk.split(X, target, groups=grp):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=500, class_weight='balanced'))
            model.fit(X[a], target[a])
            oof_probs[inds[b]] = model.predict_proba(X[b])
            oof_preds[b] = model.predict(X[b])
            
        bacc = balanced_accuracy_score(target, oof_preds)
        chance = 1.0 / len(cols)
        reliability = np.clip((bacc - chance) / 0.15, 0.0, 1.0)
        
        # Apply to validation/test passages with margin gating
        old_logits = out[:, cols].copy()
        old_p = softmax(old_logits, axis=1)
        part = np.partition(old_p, -2, axis=1)
        margin = part[:, -1] - part[:, -2]
        gate = np.clip((margin_thresh - margin) / margin_thresh, 0.0, 1.0)
        
        strength = (max_alpha * reliability) * gate
        style_p = np.maximum(oof_probs, 1e-8)
        
        mix = (1.0 - strength[:, None]) * old_p + strength[:, None] * style_p
        # Conserve total probability mass of family
        out[:, cols] = logsumexp(old_logits, axis=1, keepdims=True) + np.log(np.maximum(mix, 1e-12))
        
        print(f"  Cluster '{fname:12s}': OOF Bacc = {bacc*100:5.2f}% (Chance={chance*100:4.1f}%, Rel={reliability:4.2f})", flush=True)
        
    return out

best_score = acc_init
best_alpha = 0
for alpha in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
    for mthresh in [0.40, 0.55, 0.70]:
        refined_logits = apply_conserving_residual(base_logits, max_alpha=alpha, margin_thresh=mthresh)
        sc = eval_matching(refined_logits)
        if sc > best_score:
            best_score = sc
            best_alpha = (alpha, mthresh)
            print(f"🔥 NEW BEST CV: {best_score*100:.2f}% (alpha={alpha}, margin={mthresh})", flush=True)

print("\n" + "="*60)
print(f"🏆 CONSERVING RESIDUAL OPTIMIZED SCORE: {best_score*100:.2f}% with params {best_alpha}")
print("="*60, flush=True)

