import numpy as np
from scipy.optimize import linear_sum_assignment as _lsa
from scipy.special import softmax as _v38sp

LABELS = [f"model_{i}" for i in range(18)]
N = 36
Gte = np.array([0]*18 + [1]*18)

np.random.seed(42)
_safe_logits = np.random.randn(N, 18)
# Make row 0, 1 strong anchors
_safe_logits[0, 5] += 5.0
_safe_logits[1, 10] += 5.0

_safe_probs = _v38sp(_safe_logits, axis=1)
_v62_preds = _safe_logits.argmax(axis=1).copy()
_cluster_sizes = np.bincount(Gte)[Gte]
_dense_clusters = np.unique(Gte[(_cluster_sizes >= 12) & (_cluster_sizes <= 18)])
_n_anchors_locked = 0
_n_hungarian_reassigned = 0

for _g in _dense_clusters:
    _idx = np.where(Gte == _g)[0]
    _gsize = len(_idx)
    if _gsize <= 1:
        continue
    _sub_p = _safe_probs[_idx]
    _sub_logits = _safe_logits[_idx]
    _sorted_p = np.sort(_sub_p, axis=1)
    _margins = _sorted_p[:, -1] - _sorted_p[:, -2]
    _top_p = _sorted_p[:, -1]
    _top_cls = _sub_logits.argmax(axis=1)

    _is_anchor = (_margins >= 0.20) & (_top_p >= 0.35)
    _assigned_classes = set()
    _anchor_indices = []

    for _local_idx in np.argsort(-_top_p):
        if _is_anchor[_local_idx]:
            _cls = _top_cls[_local_idx]
            if _cls not in _assigned_classes:
                _assigned_classes.add(_cls)
                _anchor_indices.append(_local_idx)
                _n_anchors_locked += 1

    _unassigned_indices = [i for i in range(_gsize) if i not in _anchor_indices]
    if _unassigned_indices:
        _avail_classes = [c for c in range(len(LABELS)) if c not in _assigned_classes]
        if len(_avail_classes) >= len(_unassigned_indices):
            _cost = -_sub_logits[_unassigned_indices][:, _avail_classes]
            _row_ind, _col_ind = _lsa(_cost)
            for _r, _c in zip(_row_ind, _col_ind):
                _chosen = _avail_classes[_c]
                if _v62_preds[_idx[_unassigned_indices[_r]]] != _chosen:
                    _n_hungarian_reassigned += 1
                _v62_preds[_idx[_unassigned_indices[_r]]] = _chosen

print(f'Locked {_n_anchors_locked} anchors, reassigned {_n_hungarian_reassigned} items across {len(_dense_clusters)} clusters.')
assert len(_v62_preds) == N
assert set(_v62_preds[:18]) == set(range(18)) # Perfect 1-1 assignment in group 0
assert set(_v62_preds[18:]) == set(range(18)) # Perfect 1-1 assignment in group 1
print('Unit test passed with flying colors!')

