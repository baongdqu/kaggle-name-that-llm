import json, os, re

nb_path = 'champion_v90/champion-v90.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Title cell
nb['cells'][0]['source'] = [
    "# Name That LLM Champion V91 Judith-Sovereign Apex\n",
    "### Complete Multi-Model Deep Learning + Centered Dual-SVC + 54-Forensic Micro-Gates\n",
    "Author: Bao Nguyen Duy Quoc | Kaggle Name That LLM"
]

# Find cell 6 (infer.py)
code_cells = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
infer_idx = None
for i in code_cells:
    src = ''.join(nb['cells'][i]['source'])
    if 'writefile infer.py' in src:
        infer_idx = i
        break

print(f"Found infer.py at cell index: {infer_idx}")

src = ''.join(nb['cells'][infer_idx]['source'])

# Replace the V90 Sibling Meta-Learner block with V91 Forensic Meta-Learner
old_block_start = src.find("# --- V90: Pairwise Sibling Meta-Learner Apex")
old_block_end = src.find("log(f\"V90 Sibling Meta-Learner: refined {_n_tier2_fixed} sibling predictions\")")
old_block_end = src.find("\n", old_block_end) + 1

print(f"Old block range: {old_block_start} to {old_block_end}")

new_v91_block = """    # --- V91: Forensic Sibling Meta-Learner Apex (54-Dimensional Feature Matrix) ---
    import itertools as _itertools
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.svm import LinearSVC as _LSVC

    def _extract_sty_feats(_texts):
        _feats = []
        _puncs = [':', ';', '—', '–', '-', '“', '”', '’', '‘', '(', ')', '/', '!', '?', '`', '*', '"', "'"]
        for _t in _texts:
            _words = str(_t).split()
            _wlen = float(len(_words))
            _clen = float(len(str(_t)))
            _row = [_wlen, _clen / max(1.0, _wlen)]
            for _p in _puncs:
                _c = float(str(_t).count(_p))
                _row.append(_c)
                _row.append(_c / (_wlen + 1.0))
                _row.append(1.0 if _c > 0 else 0.0)
            _caps = float(len(re.findall(r'[A-Z]', str(_t))))
            _row.append(_caps / (_wlen + 1.0))
            _sents = float(len(re.findall(r'[.!?]+', str(_t))))
            _row.append(_sents)
            _row.append(_wlen / (_sents + 1.0))
            _feats.append(_row)
        return np.array(_feats, dtype=np.float32)

    _FAMILY_GROUPS = {
        'claude_opus': ['claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5'],
        'gpt_56': ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra'],
        'gemini_flash': ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'],
        'glm': ['glm-5.3', 'glm-5.3-flash']
    }
    _LAMBDA_WEIGHTS = {'claude_opus': 1.80, 'gpt_56': 0.80, 'gemini_flash': 0.60, 'glm': 0.75}

    # Discover and attach external clean passages for Opus if available
    _ext_opus_df = None
    for _p in glob.glob('/kaggle/input/**/ntl*claude*.csv', recursive=True) + glob.glob('/kaggle/input/**/*passages*.csv', recursive=True):
        try:
            _df = pd.read_csv(_p)
            if 'model' in _df.columns and 'text' in _df.columns and set(_FAMILY_GROUPS['claude_opus']).issubset(set(_df['model'].unique())):
                _ext_opus_df = _df[_df['model'].isin(_FAMILY_GROUPS['claude_opus'])].copy()
                log(f'V91: Discovered clean external Claude dataset at {_p}: {len(_ext_opus_df)} samples')
                break
        except Exception:
            pass

    _fam_clfs = {}
    _fam_means = {}
    _fam_stds = {}
    for _fname, _sibs in _FAMILY_GROUPS.items():
        if _fname == 'claude_opus' and _ext_opus_df is not None:
            _sub_tr = pd.concat([train[train['model'].isin(_sibs)], _ext_opus_df[['model', 'text']]], ignore_index=True)
        else:
            _sub_tr = train[train['model'].isin(_sibs)].reset_index(drop=True)
            
        _X_tr = _extract_sty_feats(_sub_tr['text'].values)
        _m = _X_tr.mean(axis=0); _s = _X_tr.std(axis=0) + 1e-6
        _X_tr_norm = (_X_tr - _m) / _s
        _y_tr = np.array([_sibs.index(m_) for m_ in _sub_tr['model'].values])
        
        _clf = _LR(C=0.5, max_iter=1000, random_state=42)
        _clf.fit(_X_tr_norm, _y_tr)
        _fam_clfs[_fname] = _clf
        _fam_means[_fname] = _m
        _fam_stds[_fname] = _s
    log('V91 Forensic Specialists: trained 4 intra-family 54-dimensional stylometric models')

    # Extract test features
    _test_sty_raw = _extract_sty_feats(test.text.values)
    _test_fam_logprobs = {}
    for _fname, _clf in _fam_clfs.items():
        _test_norm = (_test_sty_raw - _fam_means[_fname]) / _fam_stds[_fname]
        _test_fam_logprobs[_fname] = _clf.predict_log_proba(_test_norm)

    _n_tier2_fixed = 0
    for _g in _dense_clusters:
        _idx = np.where(Gte == _g)[0]
        if len(_idx) < 2:
            continue
        _g_preds = _v62_preds[_idx]
        _g_pred_labels = [LABELS[int(_p)] for _p in _g_preds]

        for _fam_name, _siblings in _FAMILY_GROUPS.items():
            _sib_indices = [i for i, l in enumerate(_g_pred_labels) if l in _siblings]
            if len(_sib_indices) >= 2 and len(_sib_indices) <= len(_siblings):
                _cand_classes = [L2I[s] for s in _siblings[:len(_sib_indices)]]
                _k = len(_sib_indices)
                _cost_sub = np.zeros((_k, _k), dtype=np.float64)
                _lam = _LAMBDA_WEIGHTS.get(_fam_name, 0.5)

                for _r_i, _row_idx in enumerate([_idx[ii] for ii in _sib_indices]):
                    for _c_j, _cls_idx in enumerate(_cand_classes):
                        _cls_name = LABELS[_cls_idx]
                        _sib_idx_in_fam = _siblings.index(_cls_name)
                        _base_lp = z[_row_idx, _cls_idx]
                        _sty_lp = _test_fam_logprobs[_fam_name][_row_idx, _sib_idx_in_fam]
                        _cost_sub[_r_i, _c_j] = -(_base_lp + _lam * _sty_lp)

                _r_opt, _c_opt = linear_sum_assignment(_cost_sub)
                for _r_i, _c_j in zip(_r_opt, _c_opt):
                    _target_row = _idx[_sib_indices[_r_i]]
                    _target_cls = _cand_classes[_c_j]
                    if _v62_preds[_target_row] != _target_cls:
                        _v62_preds[_target_row] = _target_cls
                        _n_tier2_fixed += 1

    log(f"V91 Forensic Meta-Learner: refined {_n_tier2_fixed} sibling predictions")
"""

src_updated = src[:old_block_start] + new_v91_block + src[old_block_end:]
nb['cells'][infer_idx]['source'] = [src_updated]

out_nb_path = 'champion_v91/champion-v91.ipynb'
with open(out_nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"✅ Successfully written updated notebook to: {out_nb_path}")

