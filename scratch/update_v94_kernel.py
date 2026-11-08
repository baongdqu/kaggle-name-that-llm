import json
import py_compile
import re

nb_path = 'champion_v94/champion-v94.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Cell 12
cell12_src = ''.join(nb['cells'][12]['source'])
cell12_src = cell12_src.replace("'--adapt_thr', '0.15'", "'--adapt_thr', '0.20'")
nb['cells'][12]['source'] = [cell12_src]

# 2. Update Cell 6 (infer.py)
cell6_src = ''.join(nb['cells'][6]['source'])

# Check target 1: capped_agglo threshold
old_agglo = "Gte = capped_agglo((Xc @ Xc.T).toarray(), 18, a.adapt_thr if a.perpassage else 0.02)"
new_agglo = "Gte = capped_agglo((Xc @ Xc.T).toarray(), 18, a.adapt_thr)"
assert old_agglo in cell6_src, "Target 1 (capped_agglo) not found!"
cell6_src = cell6_src.replace(old_agglo, new_agglo)

# Check target 2: cluster size threshold
old_dense = "_dense_clusters = np.unique(Gte[(_cluster_sizes >= 5) & (_cluster_sizes <= 18)])"
new_dense = "_dense_clusters = np.unique(Gte[(_cluster_sizes >= 2) & (_cluster_sizes <= 18)])"
assert old_dense in cell6_src, "Target 2 (dense_clusters) not found!"
cell6_src = cell6_src.replace(old_dense, new_dense)

# Check target 3: line 866 sibling candidate mapping and line 875 undefined z fix
old_sib_block = """                _cand_classes = [L2I[s] for s in _siblings[:len(_sib_indices)]]
                _k = len(_sib_indices)
                _cost_sub = np.zeros((_k, _k), dtype=np.float64)
                _lam = _LAMBDA_WEIGHTS.get(_fam_name, 0.5)

                for _r_i, _row_idx in enumerate([_idx[ii] for ii in _sib_indices]):
                    for _c_j, _cls_idx in enumerate(_cand_classes):
                        _cls_name = LABELS[_cls_idx]
                        _sib_idx_in_fam = _siblings.index(_cls_name)
                        _base_lp = z[_row_idx, _cls_idx]"""

new_sib_block = """                _cand_classes = [L2I[_g_pred_labels[i]] for i in _sib_indices]
                if len(set(_cand_classes)) < len(_sib_indices):
                    _sub_base_lps = _safe_logits[_idx[_sib_indices]][:, [L2I[s] for s in _siblings]]
                    _best_order = np.argsort(-_sub_base_lps.max(axis=0))
                    _cand_classes = [L2I[_siblings[k]] for k in _best_order[:len(_sib_indices)]]
                _k = len(_sib_indices)
                _cost_sub = np.zeros((_k, _k), dtype=np.float64)
                _lam = _LAMBDA_WEIGHTS.get(_fam_name, 0.5)

                for _r_i, _row_idx in enumerate([_idx[ii] for ii in _sib_indices]):
                    for _c_j, _cls_idx in enumerate(_cand_classes):
                        _cls_name = LABELS[_cls_idx]
                        _sib_idx_in_fam = _siblings.index(_cls_name)
                        _base_lp = _safe_logits[_row_idx, _cls_idx]"""

assert old_sib_block in cell6_src, "Target 3 (sibling block) not found!"
cell6_src = cell6_src.replace(old_sib_block, new_sib_block)

nb['cells'][6]['source'] = [cell6_src]

# Save updated notebook
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Champion V94 notebook updated successfully!")

# Extract infer.py and compile to verify zero syntax errors
infer_code = cell6_src.replace('%%writefile infer.py\n', '')
with open('scratch/v94_infer_test.py', 'w', encoding='utf-8') as f:
    f.write(infer_code)

py_compile.compile('scratch/v94_infer_test.py', doraise=True)
print("py_compile PASSED: v94_infer_test.py compiled cleanly with zero syntax/indentation errors!")
