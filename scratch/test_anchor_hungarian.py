import numpy as np
from scipy.optimize import linear_sum_assignment

K = 18
np.random.seed(42)

# Mock test set of 30 items with 2 prompt groups (sizes 15 and 15)
groups = np.array([0]*15 + [1]*15)
gsize = np.array([15]*30)
P = np.random.dirichlet(np.ones(K), size=30) * 0.1
# Make rows 0, 1, 2 strong anchors
P[0, 5] += 0.8; P[0] /= P[0].sum()
P[1, 10] += 0.8; P[1] /= P[1].sum()
P[2, 12] += 0.8; P[2] /= P[2].sum()

score = np.log(P + 1e-12)

n_hung = 0
n_anchors = 0
for g in np.unique(groups[(gsize >= 12) & (gsize <= 18)]):
    ii = np.where(groups == g)[0]
    g_size = len(ii)
    sub_score = score[ii].copy()
    sub_P = P[ii]
    
    sorted_p = np.sort(sub_P, axis=1)
    margins = sorted_p[:, -1] - sorted_p[:, -2]
    top_p = sorted_p[:, -1]
    top_cls = sub_score.argmax(axis=1)
    
    is_anchor = (margins >= 0.35) & (top_p >= 0.50)
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
                score[ii[unassigned_local[r]]] = -1e9
                score[ii[unassigned_local[r]], chosen_cls] = 0.0
        else:
            r, c = linear_sum_assignment(-sub_score)
            n_hung += int((top_cls[r] != c).sum())
            score[ii[r]] = -1e9; score[ii[r], c] = 0.0
            
    for idx_local in anchor_local:
        c = top_cls[idx_local]
        score[ii[idx_local]] = -1e9
        score[ii[idx_local], c] = 0.0

final_preds = score.argmax(axis=1)
print(f'Done! Anchors locked: {n_anchors}, Hungarian reassignments: {n_hung}')
print('Group 0 preds unique count:', len(set(final_preds[:15])))
print('Group 1 preds unique count:', len(set(final_preds[15:])))
assert len(set(final_preds[:15])) == 15, 'All 15 items in group 0 should be uniquely assigned!'
assert len(set(final_preds[15:])) == 15, 'All 15 items in group 1 should be uniquely assigned!'
assert final_preds[0] == 5, 'Anchor 0 should be locked to class 5!'
assert final_preds[1] == 10, 'Anchor 1 should be locked to class 10!'
assert final_preds[2] == 12, 'Anchor 2 should be locked to class 12!'
print('ALL ASSERTIONS PASSED PERFECTLY!')
