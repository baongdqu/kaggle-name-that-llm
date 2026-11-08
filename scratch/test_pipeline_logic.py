import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

LABELS = ['claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
 'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash','glm-5.3','glm-5.3-flash',
 'gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra','kimi-k3','qwen3.8-max']
K = len(LABELS)

# Simulate 36 test rows with 2 prompt clusters of size 18
np.random.seed(42)
N = 36
Gte = np.repeat([0, 1], 18)
test_texts = [
    "First, Step 1: Let us calculate this.",
    "The character '—' is an em-dash. Furthermore;",
    "We have \"double quotes\" in this text.",
    "Normal text without special characters."
] * 9

test = pd.DataFrame({
    'id': [f"id_{i}" for i in range(N)],
    'text': test_texts
})

# Mock logits from ensemble
raw_logits = np.random.randn(N, K) * 2.0
P = np.exp(raw_logits - raw_logits.max(1, keepdims=True))
P /= P.sum(1, keepdims=True)

# 1. Hepta-Gates
idx_g37 = LABELS.index('gemini-3.7-flash')
idx_g38 = LABELS.index('gemini-3.8-flash')
idx_glm = LABELS.index('glm-5.3')
idx_glmf = LABELS.index('glm-5.3-flash')
idx_cl_op5 = LABELS.index('claude-opus-5')
idx_cl_op46 = LABELS.index('claude-opus-4-6')
idx_gpt55 = LABELS.index('gpt-5.5')
idx_astra = LABELS.index('gpt-6-astra')
idx_terra = LABELS.index('gpt-5.6-terra')
idx_luna = LABELS.index('gpt-5.6-luna')
idx_sol = LABELS.index('gpt-5.6-sol')
idx_kimi = LABELS.index('kimi-k3')
idx_qwen = LABELS.index('qwen3.8-max')

g_boosts = 0
for i, txt in enumerate(test.text.values):
    if any(m in txt for m in ['First,', 'Second,', 'Step 1:', '1. ', 'In conclusion,']):
        if P[i, idx_g38] > 0.04:
            P[i, idx_g38] *= 1.12; P[i, idx_g37] *= 0.90; g_boosts += 1
    if any(m in txt for m in ['，', '。', '「', '」', '、', '《', '》']):
        if P[i, idx_glm] > 0.04:
            P[i, idx_glm] *= 1.15; P[i, idx_glmf] *= 0.88; g_boosts += 1
    if ' — ' in txt:
        if P[i, idx_cl_op5] > 0.04 or P[i, idx_cl_op46] > 0.04:
            if txt.count(';') >= 1 or txt.count(':') >= 2:
                P[i, idx_cl_op5] *= 1.12; P[i, idx_cl_op46] *= 0.90; g_boosts += 1
            else:
                P[i, idx_cl_op46] *= 1.10; P[i, idx_cl_op5] *= 0.92; g_boosts += 1
    if '"' in txt:
        if P[i, idx_gpt55] > 0.04:
            P[i, idx_gpt55] *= 1.15; g_boosts += 1
    words = len(txt.split())
    if words >= 50:
        comma_rate = txt.count(',') / words * 100
        if comma_rate < 7.5 and P[i, idx_astra] > 0.04:
            P[i, idx_astra] *= 1.12; P[i, idx_terra] *= 0.92; P[i, idx_luna] *= 0.92; g_boosts += 1
        elif comma_rate > 12.0 and P[i, idx_astra] > 0.04:
            P[i, idx_astra] *= 0.88; P[i, idx_terra] *= 1.08; g_boosts += 1
    if '—' in txt or txt.count(';') >= 1:
        if P[i, idx_kimi] > 0.04:
            P[i, idx_kimi] *= 1.14; P[i, idx_qwen] *= 0.88; g_boosts += 1
    elif '—' not in txt and ';' not in txt:
        if P[i, idx_qwen] > 0.04:
            P[i, idx_qwen] *= 1.10; P[i, idx_kimi] *= 0.92; g_boosts += 1
    if ('“' in txt or '”' in txt) and P[i, idx_sol] > 0.04:
        P[i, idx_sol] *= 1.12; P[i, idx_terra] *= 0.92; g_boosts += 1
P /= P.sum(1, keepdims=True)
print(f"Hepta-gates applied: {g_boosts} boosts")

# 2. Sinkhorn
Q = P.copy()
_rho = 0.35
_pred_marg = np.clip(P.mean(0), 1e-6, None); _pred_marg /= _pred_marg.sum()
_train_prior = np.ones(K) / K
_target = (1.0 - _rho) * _pred_marg + _rho * _train_prior
_target = np.clip(_target, 1e-6, None); _target /= _target.sum()
for _ in range(100):
    Q *= (len(Q) * _target[None, :]) / np.clip(Q.sum(0, keepdims=True), 1e-12, None)
    Q /= np.clip(Q.sum(1, keepdims=True), 1e-12, None)
tot = (1 - 0.45) * np.log(P + 1e-12) + 0.45 * np.log(Q + 1e-12)
print("Sinkhorn completed, shape:", tot.shape)

# 3. Safe logits (mimicking typography output)
_safe_logits = tot.copy()

# 4. Adaptive Anchor-Protected Hungarian
final_score = _safe_logits.copy()
_sz = np.bincount(Gte)[Gte]
n_anchors = 0
n_hung = 0
dense_groups = np.unique(Gte[(_sz >= 12) & (_sz <= 18)])
print("Dense groups:", dense_groups)
if len(dense_groups) > 0:
    DELTA_THRESHOLDS = np.full(K, 0.28)
    P_THRESHOLDS = np.full(K, 0.45)
    for name in ['gpt-5.5', 'kimi-k3', 'gemini-3.1-pro-preview', 'claude-fable-5', 'glm-5.3-flash']:
        idx_s = LABELS.index(name)
        DELTA_THRESHOLDS[idx_s] = 0.24
        P_THRESHOLDS[idx_s] = 0.42
    for name in ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'claude-opus-4-6', 'claude-opus-5']:
        idx_s = LABELS.index(name)
        DELTA_THRESHOLDS[idx_s] = 0.34
        P_THRESHOLDS[idx_s] = 0.50

    for _g in dense_groups:
        ii = np.where(Gte == _g)[0]
        g_size = len(ii)
        sub_score = final_score[ii].copy()
        sub_p = np.exp(sub_score - sub_score.max(1, keepdims=True))
        sub_p /= sub_p.sum(1, keepdims=True)
        top_cls = sub_p.argmax(1)
        sorted_p = np.sort(sub_p, axis=1)
        margins = sorted_p[:, -1] - sorted_p[:, -2]
        top_p = sorted_p[:, -1]

        is_anchor = (margins >= DELTA_THRESHOLDS[top_cls]) & (top_p >= P_THRESHOLDS[top_cls])
        assigned_in_group = set()
        anchor_local = []

        for idx_local in np.argsort(-top_p):
            if is_anchor[idx_local]:
                c = top_cls[idx_local]
                if c not in assigned_in_group:
                    assigned_in_group.add(c)
                    anchor_local.append(idx_local)
                    n_anchors += 1

        unassigned_local = [i for i in range(g_size) if i not in anchor_local]
        if unassigned_local:
            avail_cls = [c for c in range(K) if c not in assigned_in_group]
            if len(avail_cls) >= len(unassigned_local):
                cost = -sub_score[unassigned_local][:, avail_cls]
                r_ind, c_ind = linear_sum_assignment(cost)
                for r, c in zip(r_ind, c_ind):
                    chosen_cls = avail_cls[c]
                    if top_cls[unassigned_local[r]] != chosen_cls:
                        n_hung += 1
                    final_score[ii[unassigned_local[r]]] = -1e9
                    final_score[ii[unassigned_local[r]], chosen_cls] = 0.0
            else:
                r, c = linear_sum_assignment(-sub_score)
                n_hung += int((top_cls[r] != c).sum())
                final_score[ii[r]] = -1e9; final_score[ii[r], c] = 0.0

        for idx_local in anchor_local:
            c = top_cls[idx_local]
            final_score[ii[idx_local]] = -1e9
            final_score[ii[idx_local], c] = 0.0

    print(f"Hungarian completed: locked {n_anchors} anchors, reassigned {n_hung} ambiguous rows")

pred_final = final_score.argmax(1)
_safe = pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in pred_final]})
print("Sample predictions:\n", _safe.head(10))
assert len(_safe) == N
assert set(_safe.columns) == {'id', 'model'}
print("TEST SUCCESSFUL!")

