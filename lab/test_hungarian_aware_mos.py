import sys, os, time, re
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax

train = pd.read_csv('data/train.csv')
labels = sorted(train.model.unique())
l2i = {l: i for i, l in enumerate(labels)}
y_true = np.array([l2i[m] for m in train.model.values])
prompts = train.prompt_id.values
p2i = {p: i for i, p in enumerate(np.unique(prompts))}
g_true = np.array([p2i[p] for p in prompts])
K = 18

cache = np.load('lab/cache/base_oof.npz')
base_probs = cache['probs']
evidence = cache['evidence']

wlens = np.array([len(str(t).split()) for t in train.text.values], dtype=np.float32)
char_lens = np.array([len(str(t)) for t in train.text.values], dtype=np.float32)
avg_wlens = char_lens / np.maximum(wlens, 1.0)

LEN_BIAS = np.array([-0.0306, -0.0011, +0.1664, +0.1764, +0.0556, +0.0807,
                     -0.0362, +0.0387, -0.2429, -0.0667, +0.0950, -0.0619,
                     +0.0085, +0.0521, -0.0236, +0.0372, -0.1470, -0.1007], dtype=np.float32)
WLEN_BIAS = np.array([-0.1886, +0.3526, -0.0640, -1.1015, +0.5982, -0.0520,
                      +0.5204, +1.1509, +1.0350, -0.2390, -0.1549, -0.7586,
                      +0.1915, -0.0895, -0.1149, -0.6532, -0.3990, -0.0348], dtype=np.float32)

print("=== HUNGARIAN-AWARE MIXTURE OF SPECIALISTS (MoS) LAB ===")

SPECIALIST_GROUPS = {
    'gemini_flash': [l2i['gemini-3.6-flash'], l2i['gemini-3.7-flash'], l2i['gemini-3.8-flash']],
    'glm53': [l2i['glm-5.3'], l2i['glm-5.3-flash']],
    'gpt56': [l2i['gpt-5.6-luna'], l2i['gpt-5.6-sol'], l2i['gpt-5.6-terra']]
}

FUNCTION_WORDS = [
    'the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'for', 'it', 'with', 'as', 'was', 'on', 'at', 'by', 'this',
    'be', 'are', 'from', 'or', 'an', 'they', 'which', 'one', 'you', 'were', 'all', 'there', 'would', 'their', 'we',
    'been', 'has', 'when', 'who', 'will', 'more', 'no', 'if', 'out', 'so', 'what', 'up', 'its', 'about', 'into',
    'than', 'them', 'can', 'only', 'other', 'some', 'could', 'time', 'these', 'two', 'may', 'then', 'do', 'first',
    'any', 'now', 'such', 'like', 'over', 'even', 'most', 'also', 'after', 'did', 'many', 'before', 'must', 'through',
    'where', 'much', 'should', 'because', 'each', 'just', 'those', 'how', 'too', 'very', 'still', 'between', 'both'
]

print("Extracting micro-stylometrics...")
vec_fw = TfidfVectorizer(vocabulary=FUNCTION_WORDS, lowercase=True, norm='l2', sublinear_tf=True)
X_fw = vec_fw.fit_transform(train.text.values)

vec_char = TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', min_df=3, max_features=15000, sublinear_tf=True)
X_char = vec_char.fit_transform(train.text.values)

stats = []
for t in train.text.values:
    st = str(t)
    w = st.split()
    nw = max(1, len(w))
    sents = [s.strip() for s in re.split(r'[.!?]+', st) if s.strip()]
    slens = np.array([len(s.split()) for s in sents], dtype=np.float32)
    mslen = float(np.mean(slens))
    sslen = float(np.std(slens)) if len(slens) > 1 else 0.0
    sklen = float(np.mean(((slens - mslen) / max(sslen, 1e-4)) ** 3)) if sslen > 1e-4 else 0.0
    comma = st.count(',') / nw
    semi = st.count(';') / nw
    colon = st.count(':') / nw
    emdash = (st.count('—') + st.count('–')) / nw
    paren = st.count('(') / nw
    quote = (st.count('"') + st.count("'") + st.count('“') + st.count('”')) / nw
    hyphen = st.count('-') / nw
    awlen = len(st) / nw
    stats.append([mslen, sslen, sklen, comma, semi, colon, emdash, paren, quote, hyphen, awlen])

X_stats = np.array(stats, dtype=np.float32)
X_stats = (X_stats - X_stats.mean(axis=0)) / (X_stats.std(axis=0) + 1e-6)
X_full = sp.hstack([X_fw, X_char, sp.csr_matrix(X_stats)]).tocsr()

# 5-Fold GroupKFold OOF Probabilities for each specialist
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(train, y_true, groups=prompts))

specialist_oof_logodds = np.zeros((len(train), K), dtype=np.float64)

for name, group in SPECIALIST_GROUPS.items():
    mask = np.isin(y_true, group)
    y_group = np.array([group.index(y) if y in group else -1 for y in y_true])
    for fold, (tr_idx, va_idx) in enumerate(folds):
        tr_m = tr_idx[np.isin(y_true[tr_idx], group)]
        va_m = va_idx[np.isin(y_true[va_idx], group)]
        if len(va_m) == 0: continue
        clf = LogisticRegression(C=1.0, max_iter=300, random_state=42 + fold)
        clf.fit(X_full[tr_m], y_group[tr_m])
        probs = clf.predict_proba(X_full[va_idx])
        log_p = np.log(probs + 1e-9)
        log_p -= log_p.mean(axis=1, keepdims=True)
        for local_i, global_c in enumerate(group):
            specialist_oof_logodds[va_idx, global_c] = log_p[:, local_i]

print("OOF Specialist Log-odds computed across all 5 folds.")

def run_mos_pipeline(w_spec=0.0, use_gpt=False):
    p = base_probs.copy()
    g_sim = g_true.copy()
    w_len = 0.20
    w_wlen = 0.05
    min_wlen_size = 6
    
    # 1. Length & Word-length calibration
    for g in np.unique(g_sim):
        i_g = np.where(g_sim == g)[0]
        if len(i_g) > 1:
            mean_w = np.mean(wlens[i_g])
            std_w = np.std(wlens[i_g])
            if std_w > 1.0:
                z_w = (wlens[i_g] - mean_w) / std_w
                p[i_g] += w_len * (z_w[:, None] * LEN_BIAS[None, :])
            if w_wlen > 0 and len(i_g) >= min_wlen_size:
                mean_wl = np.mean(avg_wlens[i_g])
                std_wl = np.std(avg_wlens[i_g])
                if std_wl > 0.05:
                    z_wl = (avg_wlens[i_g] - mean_wl) / std_wl
                    p[i_g] += w_wlen * (z_wl[:, None] * WLEN_BIAS[None, :])
                    
    p_prob = np.clip(p, 1e-12, 1.0)
    p_prob /= p_prob.sum(axis=1, keepdims=True)
    
    # Soft dispersion
    p_mod = p_prob.copy()
    for g in np.unique(g_sim):
        i_g = np.where(g_sim == g)[0]
        if len(i_g) > 1:
            for c in range(K):
                c_rows = i_g[p_mod[i_g].argmax(axis=1) == c]
                if len(c_rows) > 1:
                    confs = p_mod[c_rows, c]
                    ord_c = np.argsort(-confs)
                    for loser in c_rows[ord_c[1:]]:
                        p_mod[loser, c] *= 0.65
                    p_mod[c_rows] /= (p_mod[c_rows].sum(axis=1, keepdims=True) + 1e-12)
                    
    # Damped Sinkhorn 0.45
    q = p_mod.copy()
    target = np.bincount(y_true, minlength=18).astype(np.float64) / len(y_true)
    for _ in range(60):
        q *= (len(q) * target[None, :]) / np.clip(q.sum(0, keepdims=True), 1e-12, None)
        q /= np.clip(q.sum(1, keepdims=True), 1e-12, None)
    tot = 0.55 * np.log(p_mod + 1e-12) + 0.45 * np.log(q + 1e-12)
    
    # === PRE-HUNGARIAN SPECIALIST LOGIT INJECTION ===
    if w_spec > 0:
        target_groups = ['gemini_flash', 'glm53'] + (['gpt56'] if use_gpt else [])
        for name in target_groups:
            group = SPECIALIST_GROUPS[name]
            # Sample where top prediction is in group
            top_is_group = np.isin(tot.argmax(axis=1), group)
            for c in group:
                tot[top_is_group, c] += w_spec * specialist_oof_logodds[top_is_group, c]
                
    # Hungarian Assignment inside dense clusters
    pred = tot.argmax(axis=1).copy()
    c_sizes = np.bincount(g_sim)[g_sim]
    dense = np.unique(g_sim[(c_sizes >= 5) & (c_sizes <= 18)])
    probs_tot = softmax(tot, axis=1)
    
    for g in dense:
        i_g = np.where(g_sim == g)[0]
        if len(i_g) <= 1: continue
        sub_p = probs_tot[i_g]
        sub_log = tot[i_g]
        sort_p = np.sort(sub_p, axis=1)
        margins = sort_p[:, -1] - sort_p[:, -2]
        top_p = sort_p[:, -1]
        top_c = sub_log.argmax(axis=1)
        
        is_anc = (margins >= 0.20) & (top_p >= 0.35)
        assigned = set(); anc_idx = []
        for l_idx in np.argsort(-top_p):
            if is_anc[l_idx]:
                cls_ = top_c[l_idx]
                if cls_ not in assigned:
                    assigned.add(cls_)
                    anc_idx.append(l_idx)
        unassigned = [i for i in range(len(i_g)) if i not in anc_idx]
        if unassigned:
            avail = [c for c in range(K) if c not in assigned]
            if len(avail) >= len(unassigned):
                cost = -sub_log[unassigned][:, avail]
                r_ind, c_ind = linear_sum_assignment(cost)
                for r, c in zip(r_ind, c_ind):
                    pred[i_g[unassigned[r]]] = avail[c]
    return pred

print("\n=== SWEEPING SPECIALIST LOGIT WEIGHT w_spec (PRE-HUNGARIAN) ===")
baseline_v68 = run_mos_pipeline(w_spec=0.0)
base_acc = balanced_accuracy_score(y_true, baseline_v68)
print(f"Baseline V68 Balanced Accuracy: {base_acc:.4f} (Errors: {(baseline_v68 != y_true).sum()})")

for use_gpt in [False, True]:
    print(f"\n--- Include GPT-5.6 Trio: {use_gpt} ---")
    for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        pr = run_mos_pipeline(w_spec=w, use_gpt=use_gpt)
        acc = balanced_accuracy_score(y_true, pr)
        n_diff = (pr != baseline_v68).sum()
        n_fixed = ((pr == y_true) & (baseline_v68 != y_true)).sum()
        n_broken = ((pr != y_true) & (baseline_v68 == y_true)).sum()
        net = n_fixed - n_broken
        print(f"w_spec={w:.2f}: Acc={acc:.4f} (Delta={acc-base_acc:+.4f}) | Diff={n_diff:3d} | Fixed={n_fixed:2d} | Broken={n_broken:2d} | Net={net:+2d}")
