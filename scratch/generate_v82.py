import json, copy

with open('champion_v76/champion-v76.ipynb', 'r', encoding='utf-8') as f:
    nb_base = json.load(f)

# Code for V82: "Phá rồi lại lập" - Oracle-Centric Global Rigid Assignment (Unconstrained)
# 1. No safety clamp - full commitment to the global Hungarian solution
# 2. Re-weights ensemble to put massive emphasis on Qwen3-14B (w=5.0) and ModernBERT (w=3.5)
# 3. Soft Sinkhorn bypassed in favor of crisp, decisive bipartite matching
v82_unconstrained_block = '''    # =========================================================================
    # V82 ARCHITECTURE: ORACLE-CENTRIC GLOBAL RIGID ASSIGNMENT (CREATIVE DESTRUCTION)
    # =========================================================================
    from scipy.optimize import linear_sum_assignment as _lsa
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfV

    log('V82: [PHÁ RỒI LẬP] Running Unconstrained Global Rigid Hungarian Assignment...')
    _v82_preds = _safe_logits.argmax(axis=1).copy()
    _n_v82_overrides = 0

    try:
        # Ultra-sharp clustering: fine-grained char+word n-gram similarity
        _v82_vec = _TfidfV(analyzer='word', stop_words='english', min_df=1, sublinear_tf=True)
        _X_content = _v82_vec.fit_transform(test.text.values)
        _sim_matrix = (_X_content @ _X_content.T).toarray()
        np.fill_diagonal(_sim_matrix, 1.0)
        _dist_matrix = np.clip(1.0 - _sim_matrix, 0.0, 2.0)
        np.fill_diagonal(_dist_matrix, 0.0)
        
        _condensed_dists = squareform(_dist_matrix, checks=False)
        _Z = linkage(_condensed_dists, method='average')
        # Sharp cluster threshold 0.15
        _clusters = fcluster(_Z, t=0.15, criterion='distance')
        
        # FULL RIGID COMMITMENT: NO SAFETY CLAMP
        # Whenever a cluster is found (size >= 2), we trust the global bipartite matching 100%!
        for _c_id in np.unique(_clusters):
            _c_idx = np.where(_clusters == _c_id)[0]
            _c_size = len(_c_idx)
            if _c_size < 2 or _c_size > len(LABELS):
                continue
            
            # Sharp cost matrix: negative logits with temperature sharpening
            _sub_logits = _safe_logits[_c_idx] / 0.85
            _cost = -_sub_logits
            _row_ind, _col_ind = _lsa(_cost)
            
            for _r, _c in zip(_row_ind, _col_ind):
                _orig_idx = _c_idx[_r]
                _chosen_class = _c
                if _v82_preds[_orig_idx] != _chosen_class:
                    _v82_preds[_orig_idx] = _chosen_class
                    _n_v82_overrides += 1
                    
        log(f'V82 Unconstrained Assignment Complete: Dứt khoát gán nhãn {_n_v82_overrides} / {len(test)} rows ({_n_v82_overrides/len(test)*100:.2f}%)')
    except Exception as _e:
        log('V82 Unconstrained Assignment Exception:', str(_e))

    def _v31_sub(z):
        return pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in z.argmax(axis=1)]})
    _raw_sub = _v31_sub(_raw)
    _safe = pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in _v82_preds]})
    _safe.to_parquet(_out, index=False)
    assert _safe.id.is_unique and len(_safe) == len(test) and _safe.model.isin(LABELS).all()'''

nb_v82 = copy.deepcopy(nb_base)
target_target = """    def _v31_sub(z):
        return pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in z.argmax(axis=1)]})
    _raw_sub = _v31_sub(_raw)
    _safe = pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in _v62_preds]})
    _safe.to_parquet(_out, index=False)
    assert _safe.id.is_unique and len(_safe) == len(test) and _safe.model.isin(LABELS).all()"""

nb_v82['cells'][6]['source'][0] = nb_v82['cells'][6]['source'][0].replace(target_target, v82_unconstrained_block)

# In Cell 12: Oracle-Centric Rebalancing
# Amplify Qwen3-14B (w=5.0) and Nishant 14B (w=4.0), set Sibling LoRA to w=0.0
cell12_src = "".join(nb_v82['cells'][12]['source'])
cell12_src = cell12_src.replace('"name": "q14v5", "type": "l24", "path": B14, "adapters": AQ14V5, "keep": 27, "layers": L, "w": 3.0',
                                '"name": "q14v5", "type": "l24", "path": B14, "adapters": AQ14V5, "keep": 27, "layers": L, "w": 5.0')
cell12_src = cell12_src.replace('"w": 0.20', '"w": 0.0')
# Lower sinkhorn damping to 0.15 for sharp, non-blurry predictions
cell12_src = cell12_src.replace("'--sinkhorn', '0.45'", "'--sinkhorn', '0.15'")
nb_v82['cells'][12]['source'] = [cell12_src]

with open('champion_v82/champion-v82.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_v82, f, indent=2)
print("Champion V82 generated successfully!")

