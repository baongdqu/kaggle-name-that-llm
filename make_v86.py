import json

nb = json.load(open('champion_v85/champion-v85.ipynb', encoding='utf-8'))
src = ''.join(nb['cells'][6]['source'])

tier2_code = '''
    # --- V86: Two-Stage Hierarchical Sibling Disentanglement (Intra-Family Permutation Fusion) ---
    import itertools as _itertools
    _FAMILY_GROUPS = {
        'claude_opus': ['claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5'],
        'gpt_56': ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra'],
        'gemini_flash': ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash']
    }
    _LAMBDA_WEIGHTS = {'claude_opus': 1.0, 'gpt_56': 0.40, 'gemini_flash': 0.25}
    _n_tier2_fixed = 0

    _test_wlens = np.array([len(str(_t).split()) for _t in test.text.values], dtype=np.float32)
    _test_em = np.array([str(_t).count('—') for _t in test.text.values], dtype=np.float32)
    _test_semi = np.array([str(_t).count(';') for _t in test.text.values], dtype=np.float32)
    _test_colon = np.array([str(_t).count(':') for _t in test.text.values], dtype=np.float32)

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
                _lam = _LAMBDA_WEIGHTS.get(_fam_name, 0.3)

                _g_wlens = _test_wlens[_sib_rows]
                _std_w = _g_wlens.std()
                _rel_w = (_g_wlens - _g_wlens.mean()) / max(1.0, _std_w)

                _combined_scores = np.zeros((_k, _k), dtype=np.float32)
                for _r_i in range(_k):
                    _row_idx = _sib_rows[_r_i]
                    for _c_j, _cls in enumerate(_cand_classes):
                        _cls_name = LABELS[_cls]
                        _t1 = _safe_logits[_row_idx, _cls]
                        _sty = 0.0
                        if 'opus' in _cls_name:
                            if _cls_name == 'claude-opus-5': _sty += 0.30 * _test_colon[_row_idx] + 0.20 * _test_semi[_row_idx]
                            elif _cls_name == 'claude-opus-4-8': _sty += 0.20 * _rel_w[_r_i] - 0.20 * _test_em[_row_idx]
                            elif _cls_name == 'claude-opus-4-6': _sty += 0.25 * _test_em[_row_idx] - 0.15 * _rel_w[_r_i]
                        elif 'gpt-5.6' in _cls_name:
                            if _cls_name == 'gpt-5.6-sol': _sty += 0.25 * _test_semi[_row_idx] + 0.15 * _rel_w[_r_i]
                            elif _cls_name == 'gpt-5.6-luna': _sty += 0.20 * _test_em[_row_idx]
                            elif _cls_name == 'gpt-5.6-terra': _sty -= 0.15 * _rel_w[_r_i]
                        elif 'gemini' in _cls_name:
                            if _cls_name == 'gemini-3.7-flash': _sty += 0.20 * _rel_w[_r_i]
                            elif _cls_name == 'gemini-3.8-flash': _sty -= 0.25 * _rel_w[_r_i]
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

    log(f"V86 Two-Stage Sibling Disentanglement: refined {_n_tier2_fixed} sibling predictions")
'''

anchor_str = "_safe = pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in _v62_preds]})"
if anchor_str in src:
    new_src = src.replace(anchor_str, tier2_code + '\n    ' + anchor_str)
    nb['cells'][6]['source'] = [l + '\n' for l in new_src.splitlines()]
    
    # Also update notebook title/comment in cell 0
    nb['cells'][0]['source'] = ['# Name That LLM Champion V86 Two-Stage Sibling Fusion\n']
    
    with open('champion_v86/champion-v86.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("SUCCESS: champion_v86/champion-v86.ipynb built successfully!")
else:
    print("ERROR: anchor_str not found in infer.py!")

