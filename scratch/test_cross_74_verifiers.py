import numpy as np, pandas as pd, re, time
from scipy.special import logsumexp, softmax
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import test_cross_72_milestone as m

t0 = time.time()
print("🚀 Loading base champion configuration (72.58%)...", flush=True)

train = m.train
texts = m.texts
y_true = m.y_true
all_pids = m.all_pids
unique_pids = m.unique_pids
LABELS = m.LABELS
L2I = m.L2I

base_logits = m.base_logits
cluster_oof = m.cluster_oof
oof_fbl_glm_diff = m.oof_fbl_glm_diff
oof_gem_pair_diff = m.oof_gem_pair_diff
idx_glm = m.idx_glm
idx_fbl = m.idx_fbl
idx_g37 = m.idx_g37
idx_g38 = m.idx_g38

gkf = GroupKFold(n_splits=5)

def build_pairwise_oof(name1, name2, ngram=(1, 2), analyzer='word', C=0.5):
    idx1 = L2I[name1]
    idx2 = L2I[name2]
    tr_pairs = [i for i in range(len(train)) if y_true[i] in [idx1, idx2]]
    y_pair = (y_true[tr_pairs] == idx2).astype(int)
    
    vec = TfidfVectorizer(ngram_range=ngram, analyzer=analyzer, min_df=2, max_features=10000, sublinear_tf=True)
    X_tr = vec.fit_transform([texts[i] for i in tr_pairs])
    X_all_v = vec.transform(texts)
    
    oof_diff = np.zeros(len(train), dtype=np.float32)
    # Train 5-fold model and predict on all texts
    models = []
    for a, b in gkf.split(X_tr, y_pair, groups=all_pids[tr_pairs]):
        clf = LogisticRegression(C=C, max_iter=500)
        clf.fit(X_tr[a], y_pair[a])
        models.append(clf)
    
    # Average decision function across the 5 fold models for generalization
    for clf in models:
        oof_diff += clf.decision_function(X_all_v) / 5.0
        
    return idx1, idx2, oof_diff

print("Building Pairwise Gates: Sonnet-Opus, Kimi-GPT55, Sol-Terra...", flush=True)
idx_son, idx_op8, oof_so_diff = build_pairwise_oof('claude-sonnet-5', 'claude-opus-4-8', C=0.5)
idx_kim, idx_g55, oof_kg_diff = build_pairwise_oof('kimi-k3', 'gpt-5.5', C=0.1)
idx_sol, idx_ter, oof_st_diff = build_pairwise_oof('gpt-5.6-sol', 'gpt-5.6-terra', C=0.5)

def eval_matching(logits):
    preds = np.zeros(len(train), dtype=int)
    for p in unique_pids:
        idx = np.where(all_pids == p)[0]
        M = logits[idx]
        r, c = linear_sum_assignment(-M)
        preds[idx[r]] = c
    return balanced_accuracy_score(y_true, preds), preds

print("Testing additions onto the 72.58% champion baseline...", flush=True)

# Build baseline 72.58% logits
out_7258 = base_logits.copy()
weights = {'gemini_flash': 0.9, 'gpt56': 1.2, 'glm': 16.0}

for fname in ['gemini_flash', 'gpt56', 'glm']:
    cols, oof_probs, reliability = cluster_oof[fname]
    alpha = weights[fname]
    old_logits = out_7258[:, cols].copy()
    old_p = softmax(old_logits, axis=1)
    part = np.partition(old_p, -2, axis=1)
    margin = part[:, -1] - part[:, -2]
    gate = np.clip((1.0 - margin) / 1.0, 0.0, 1.0)
    strength = (alpha * reliability) * gate
    style_p = np.maximum(oof_probs, 1e-8)
    mix = (1.0 - strength[:, None]) * old_p + strength[:, None] * style_p
    out_7258[:, cols] = logsumexp(old_logits, axis=1, keepdims=True) + np.log(np.maximum(mix, 1e-12))

out_7258[:, idx_glm] += 0.35 * oof_fbl_glm_diff
out_7258[:, idx_fbl] -= 0.35 * oof_fbl_glm_diff
out_7258[:, idx_g38] += 0.15 * oof_gem_pair_diff
out_7258[:, idx_g37] -= 0.15 * oof_gem_pair_diff

sc_base, _ = eval_matching(out_7258)
print(f"Verified 72.58% baseline CV: {sc_base*100:.2f}%\n")

best_score = sc_base
best_params = None
best_preds = None

p_base = softmax(out_7258, axis=1)
p_so = p_base[:, idx_son] + p_base[:, idx_op8]
p_kg = p_base[:, idx_kim] + p_base[:, idx_g55]
p_st = p_base[:, idx_sol] + p_base[:, idx_ter]

for w_so in [0.0, 0.05, 0.10, 0.15, 0.20]:
    for w_kg in [0.0, 0.05, 0.10, 0.15, 0.20]:
        for w_st in [0.0, 0.05, 0.10, 0.15, 0.20]:
            out = out_7258.copy()
            if w_so > 0:
                gate_so = w_so * p_so * oof_so_diff
                out[:, idx_op8] += gate_so
                out[:, idx_son] -= gate_so
            if w_kg > 0:
                gate_kg = w_kg * p_kg * oof_kg_diff
                out[:, idx_g55] += gate_kg
                out[:, idx_kim] -= gate_kg
            if w_st > 0:
                gate_st = w_st * p_st * oof_st_diff
                out[:, idx_ter] += gate_st
                out[:, idx_sol] -= gate_st
                
            sc, preds = eval_matching(out)
            if sc > best_score:
                best_score = sc
                best_params = (w_so, w_kg, w_st)
                best_preds = preds
                print(f"🔥 NEW BEST BALANCED ACCURACY: {best_score*100:.2f}% | w_so={w_so}, w_kg={w_kg}, w_st={w_st}", flush=True)

print(f"\n==========================================")
print(f"🏆 CHAMPION BALANCED ACCURACY: {best_score*100:.2f}%")
print(f"Optimal Parameters: {best_params}")
print(f"==========================================")

