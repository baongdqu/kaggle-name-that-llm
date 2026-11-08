import json, copy

with open('champion_v76/champion-v76.ipynb', 'r', encoding='utf-8') as f:
    nb_base = json.load(f)

# Code for V81: Enhanced Test-Internal Prompt Clustering & Hungarian Assignment with Safety Guard
# 1. Clusters test rows by prompt content TF-IDF (low threshold 0.12 to capture variations)
# 2. For dense clusters (size 2 to 18), solves Kuhn-Munkres bipartite matching
# 3. Applies strict Safety Guard: only modifies rows where Anchor Margin < 0.22
v81_internal_hungarian_block = '''    # =========================================================================
    # V81 ARCHITECTURE: ENHANCED TEST-INTERNAL PROMPT HUNGARIAN WITH SAFETY GUARD
    # =========================================================================
    from scipy.optimize import linear_sum_assignment as _lsa
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfV

    log('V81: Running Test-Internal Prompt Clustering & Hungarian Assignment...')
    _safe_probs = _v38sp.softmax(_safe_logits, axis=1)
    _sorted_p = np.sort(_safe_probs, axis=1)
    _anchor_margins = _sorted_p[:, -1] - _sorted_p[:, -2]
    _anchor_top_p = _sorted_p[:, -1]
    
    _v81_preds = _v62_preds.copy()
    _n_v81_reassigned = 0
    _n_v81_guard_preserved = 0

    try:
        # Build content-word TF-IDF representations for prompt clustering
        _v81_vec = _TfidfV(analyzer='word', stop_words='english', min_df=1, sublinear_tf=True)
        _X_content = _v81_vec.fit_transform(test.text.values)
        _sim_matrix = (_X_content @ _X_content.T).toarray()
        np.fill_diagonal(_sim_matrix, 1.0)
        _dist_matrix = np.clip(1.0 - _sim_matrix, 0.0, 2.0)
        np.fill_diagonal(_dist_matrix, 0.0)
        
        # Average-linkage clustering with strict threshold 0.12
        _condensed_dists = squareform(_dist_matrix, checks=False)
        _Z = linkage(_condensed_dists, method='average')
        _clusters = fcluster(_Z, t=0.12, criterion='distance')
        
        _cluster_sizes = np.bincount(_clusters)
        log(f'V81 Clustering: Found {len(_cluster_sizes)} clusters. Max cluster size: {_cluster_sizes.max()}')

        for _c_id in np.unique(_clusters):
            _c_idx = np.where(_clusters == _c_id)[0]
            _c_size = len(_c_idx)
            if _c_size < 2 or _c_size > len(LABELS):
                continue
            
            _sub_logits = _safe_logits[_c_idx]
            # Cost is negative log-likelihood
            _cost = -_sub_logits
            _row_ind, _col_ind = _lsa(_cost)
            
            for _r, _c in zip(_row_ind, _col_ind):
                _orig_idx = _c_idx[_r]
                _proposed_class = _c
                
                # SAFETY GUARD FLOOR:
                # If Golden Anchor is already highly confident (margin >= 0.22 and top_p >= 0.38),
                # preserve the Golden Anchor decision to lock in the 0.662 floor!
                if _anchor_margins[_orig_idx] >= 0.22 and _anchor_top_p[_orig_idx] >= 0.38:
                    _n_v81_guard_preserved += 1
                else:
                    if _v81_preds[_orig_idx] != _proposed_class:
                        _v81_preds[_orig_idx] = _proposed_class
                        _n_v81_reassigned += 1
                        
        log(f'V81 Hungarian Complete: Reassigned {_n_v81_reassigned} rows, Safety Guard preserved {_n_v81_guard_preserved} anchor rows.')
    except Exception as _e:
        log('V81 Internal Hungarian Exception (falling back safely to Anchor):', str(_e))

    def _v31_sub(z):
        return pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in z.argmax(axis=1)]})
    _raw_sub = _v31_sub(_raw)
    _safe = pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in _v81_preds]})
    _safe.to_parquet(_out, index=False)
    assert _safe.id.is_unique and len(_safe) == len(test) and _safe.model.isin(LABELS).all()'''

nb_v81 = copy.deepcopy(nb_base)
target_target = """    def _v31_sub(z):
        return pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in z.argmax(axis=1)]})
    _raw_sub = _v31_sub(_raw)
    _safe = pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in _v62_preds]})
    _safe.to_parquet(_out, index=False)
    assert _safe.id.is_unique and len(_safe) == len(test) and _safe.model.isin(LABELS).all()"""

nb_v81['cells'][6]['source'][0] = nb_v81['cells'][6]['source'][0].replace(target_target, v81_internal_hungarian_block)

# In Cell 12: Set w=0.0 for Sibling LoRA to remove noise completely
cell12_src = "".join(nb_v81['cells'][12]['source'])
cell12_src = cell12_src.replace('"w": 0.20', '"w": 0.0')
nb_v81['cells'][12]['source'] = [cell12_src]

with open('champion_v81/champion-v81.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_v81, f, indent=2)
print("Champion V81 generated successfully!")

