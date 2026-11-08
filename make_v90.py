import json

# Load base champion_v89 notebook
nb = json.load(open('champion_v89/champion-v89.ipynb', encoding='utf-8'))

# Update Title
nb['cells'][0]['source'] = ['# Name That LLM Champion V90 Pairwise Sibling Meta-Learner Apex\n']

infer_src = ''.join(nb['cells'][6]['source'])

v90_sibling_code = '''    # --- V90: Pairwise Sibling Meta-Learner Apex (Trained 9-Stylometric Logistic Regressors) ---
    import itertools as _itertools
    from sklearn.linear_model import LogisticRegression as _LR

    def _extract_sty_feats(_texts):
        _feats = []
        for _t in _texts:
            _words = str(_t).split()
            _wlen = len(_words)
            _clen = len(str(_t))
            _feats.append([
                _wlen,
                _clen / max(1, _wlen),
                str(_t).count('—'),
                str(_t).count(';'),
                str(_t).count(':'),
                str(_t).count('?'),
                str(_t).count('!'),
                str(_t).count('('),
                str(_t).count('"')
            ])
        _X = np.array(_feats, dtype=np.float32)
        return _X

    _FAMILY_GROUPS = {
        'claude_opus': ['claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5'],
        'gpt_56': ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra'],
        'gemini_flash': ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash']
    }
    _LAMBDA_WEIGHTS = {'claude_opus': 1.50, 'gpt_56': 0.60, 'gemini_flash': 0.45}

    # Train specialized 9-stylometric Logistic Regressors on official train.csv
    _fam_clfs = {}
    _fam_means = {}
    _fam_stds = {}
    for _fname, _sibs in _FAMILY_GROUPS.items():
        _sub_tr = train[train['model'].isin(_sibs)].reset_index(drop=True)
        _X_tr = _extract_sty_feats(_sub_tr['text'].values)
        _m = _X_tr.mean(axis=0); _s = _X_tr.std(axis=0) + 1e-6
        _X_tr_norm = (_X_tr - _m) / _s
        _y_tr = np.array([_sibs.index(m_) for m_ in _sub_tr['model'].values])
        _clf = _LR(C=1.0, max_iter=1000, random_state=42)
        _clf.fit(_X_tr_norm, _y_tr)
        _fam_clfs[_fname] = _clf
        _fam_means[_fname] = _m
        _fam_stds[_fname] = _s
    log('V90 Sibling Specialists: trained 3 intra-family stylometric models on official train')

    # Extract test features
    _test_sty_raw = _extract_sty_feats(test.text.values)
    _test_fam_logprobs = {}
    for _fname, _clf in _fam_clfs.items():
        _test_norm = (_test_sty_raw - _fam_means[_fname]) / _fam_stds[_fname]
        _test_fam_logprobs[_fname] = _clf.predict_log_proba(_test_norm)

    _n_tier2_fixed = 0
    for _g in _dense_clusters:
        _idx = np.where(Gte == _g)[0]
        if len(_idx) < 3:
            continue
        _g_preds = _v62_preds[_idx]
        _g_pred_labels = [LABELS[int(_p)] for _p in _g_preds]

        for _fam_name, _siblings in _FAMILY_GROUPS.items():
            _sib_indices = [i for i, l in enumerate(_g_pred_labels) if l in _siblings]
            if len(_sib_indices) >= 2 and len(_sib_indices) <= len(_siblings):
                _sib_rows = _idx[_sib_indices]
                _k = len(_sib_indices)
                _cand_classes = [L2I[s] for s in _siblings[:_k]]
                _lam = _LAMBDA_WEIGHTS.get(_fam_name, 0.4)

                _combined_scores = np.zeros((_k, _k), dtype=np.float32)
                for _r_i in range(_k):
                    _row_idx = _sib_rows[_r_i]
                    _spec_lp = _test_fam_logprobs[_fam_name][_row_idx]
                    for _c_j, _cls in enumerate(_cand_classes):
                        _cls_name = LABELS[_cls]
                        _t1 = _safe_logits[_row_idx, _cls]
                        # Blend the specialized model probability
                        _sib_idx_in_fam = _siblings.index(_cls_name)
                        _sty = _spec_lp[_sib_idx_in_fam]
                        _combined_scores[_r_i, _c_j] = _t1 + _lam * _sty

                _best_perm, _best_val = None, -1e9
                for _perm in _itertools.permutations(range(_k)):
                    _val = sum(_combined_scores[_i, _perm[_i]] for _i in range(_k))
                    if _val > _best_val:
                        _best_val, _best_perm = _val, _perm
                if _best_perm is not None:
                    for _i in range(_k):
                        _new_cls = _cand_classes[_best_perm[_i]]
                        if _v62_preds[_sib_rows[_i]] != _new_cls:
                            _n_tier2_fixed += 1
                        _v62_preds[_sib_rows[_i]] = _new_cls

    log(f"V90 Sibling Meta-Learner: refined {_n_tier2_fixed} sibling predictions")
'''

# Find V86/V89 tier2 code block in infer_src and replace it
anchor_v86 = '    # --- V86: Two-Stage Hierarchical Sibling Disentanglement (Intra-Family Permutation Fusion) ---'
anchor_end = '    _safe = pd.DataFrame({\'id\': test.id.values, \'model\': [LABELS[int(i)] for i in _v62_preds]})'

start_pos = infer_src.find(anchor_v86)
end_pos = infer_src.find(anchor_end)

if start_pos != -1 and end_pos != -1:
    infer_src = infer_src[:start_pos] + v90_sibling_code + '\n    ' + infer_src[end_pos:]
    print("SUCCESS: Replaced V89 logic with V90 Pairwise Sibling Meta-Learner code!")
else:
    print(f"ERROR: Anchors not found! start_pos={start_pos}, end_pos={end_pos}")

nb['cells'][6]['source'] = [l + '\n' for l in infer_src.splitlines()]

# Cell 12: Ensure parameters are optimized
cmd_src = ''.join(nb['cells'][12]['source'])
nb['cells'][12]['source'] = [l + '\n' for l in cmd_src.splitlines()]

with open('champion_v90/champion-v90.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("SUCCESS: champion_v90/champion-v90.ipynb created successfully!")

