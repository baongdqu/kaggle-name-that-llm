import json, re

mos_code = r'''    log(f'V62 Anchor-Hungarian: locked {_n_anchors_locked} anchors, reassigned {_n_hungarian_reassigned} ambiguous items across {len(_dense_clusters)} dense clusters')

    # =========================================================================
    # --- Champion V69: Mixture of Specialists (MoS) Selective Stage-2 Apex ---
    # Targets the exact Bayes error bottleneck (Gemini Flash trio & GLM 5.3 duo)
    # using Function Word (Burrows' Delta) + Char N-Grams (3-5) + Sentence Stats.
    # Trains in ~9s on CPU; overrides ONLY when specialist confidence >= 0.60.
    # =========================================================================
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer as _SpTfidf
        from sklearn.linear_model import LogisticRegression as _SpLR

        _FUNCTION_WORDS = [
            'the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'for', 'it', 'with', 'as', 'was', 'on', 'at', 'by', 'this',
            'be', 'are', 'from', 'or', 'an', 'they', 'which', 'one', 'you', 'were', 'all', 'there', 'would', 'their', 'we',
            'been', 'has', 'when', 'who', 'will', 'more', 'no', 'if', 'out', 'so', 'what', 'up', 'its', 'about', 'into',
            'than', 'them', 'can', 'only', 'other', 'some', 'could', 'time', 'these', 'two', 'may', 'then', 'do', 'first',
            'any', 'now', 'such', 'like', 'over', 'even', 'most', 'also', 'after', 'did', 'many', 'before', 'must', 'through',
            'where', 'much', 'should', 'because', 'each', 'just', 'those', 'how', 'too', 'very', 'still', 'between', 'both'
        ]

        def _get_micro_features(_raw_texts, _vec_fw=None, _vec_char=None, _scaler_stats=None):
            _is_fit = (_vec_fw is None)
            if _is_fit:
                _vec_fw = _SpTfidf(vocabulary=_FUNCTION_WORDS, lowercase=True, norm='l2', sublinear_tf=True)
                _xfw = _vec_fw.fit_transform(_raw_texts)
                _vec_char = _SpTfidf(ngram_range=(3, 5), analyzer='char_wb', min_df=3, max_features=15000, sublinear_tf=True)
                _xchar = _vec_char.fit_transform(_raw_texts)
            else:
                _xfw = _vec_fw.transform(_raw_texts)
                _xchar = _vec_char.transform(_raw_texts)

            _stats_list = []
            for _t in _raw_texts:
                _st = str(_t)
                _words = _st.split()
                _nw = max(1, len(_words))
                _sents = [s.strip() for s in re.split(r'[.!?]+', _st) if s.strip()]
                _slens = [len(s.split()) for s in _sents]
                _mslen = float(np.mean(_slens))
                _sslen = float(np.std(_slens)) if len(_slens) > 1 else 0.0
                _sklen = float(np.mean(((_slens - _mslen) / max(_sslen, 1e-4)) ** 3)) if _sslen > 1e-4 else 0.0
                _comma = _st.count(',') / _nw
                _semi = _st.count(';') / _nw
                _colon = _st.count(':') / _nw
                _emdash = (_st.count('—') + _st.count('–')) / _nw
                _paren = _st.count('(') / _nw
                _quote = (_st.count('"') + _st.count("'") + _st.count('“') + _st.count('”')) / _nw
                _hyphen = _st.count('-') / _nw
                _awlen = len(_st) / _nw
                _stats_list.append([_mslen, _sslen, _sklen, _comma, _semi, _colon, _emdash, _paren, _quote, _hyphen, _awlen])

            _xstats = np.array(_stats_list, dtype=np.float32)
            if _is_fit:
                _stat_mean = _xstats.mean(axis=0)
                _stat_std = _xstats.std(axis=0) + 1e-6
                _scaler_stats = (_stat_mean, _stat_std)
            else:
                _stat_mean, _stat_std = _scaler_stats

            _xstats_norm = (_xstats - _stat_mean) / _stat_std
            _xall = sp.hstack([_xfw, _xchar, sp.csr_matrix(_xstats_norm)]).tocsr()
            return _xall, _vec_fw, _vec_char, _scaler_stats

        log('V69 MoS: Extracting micro-stylometric features for Stage-2 Specialists...')
        _train_texts = train.text.values
        _test_texts = test.text.values
        _X_tr_spec, _v_fw, _v_char, _sc_stats = _get_micro_features(_train_texts)
        _X_te_spec, _, _, _ = _get_micro_features(_test_texts, _v_fw, _v_char, _sc_stats)

        _train_labels = np.array([LABELS.index(m) for m in train.model.values])
        _mos_targets = {
            'gemini_flash': [LABELS.index('gemini-3.6-flash'), LABELS.index('gemini-3.7-flash'), LABELS.index('gemini-3.8-flash')],
            'glm53': [LABELS.index('glm-5.3'), LABELS.index('glm-5.3-flash')]
        }

        _specialist_models = {}
        for _grp_name, _cls_list in _mos_targets.items():
            _mask = np.isin(_train_labels, _cls_list)
            _y_local = np.array([_cls_list.index(l) for l in _train_labels[_mask]])
            _clf = _SpLR(C=1.0, max_iter=300, random_state=42)
            _clf.fit(_X_tr_spec[_mask], _y_local)
            _specialist_models[_grp_name] = (_clf, _cls_list)
            log(f'V69 MoS: Fitted specialist [{_grp_name}] on {_mask.sum()} samples')

        _mos_override_count = 0
        for _grp_name, (_clf, _cls_list) in _specialist_models.items():
            _in_grp = np.isin(_v62_preds, _cls_list)
            _target_te_idx = np.where(_in_grp)[0]
            if len(_target_te_idx) > 0:
                _probs = _clf.predict_proba(_X_te_spec[_target_te_idx])
                _max_p = _probs.max(axis=1)
                _chosen_local = _probs.argmax(axis=1)
                for _i, _loc_idx in enumerate(_target_te_idx):
                    if _max_p[_i] >= 0.60:
                        _chosen_global = _cls_list[_chosen_local[_i]]
                        if _v62_preds[_loc_idx] != _chosen_global:
                            _v62_preds[_loc_idx] = _chosen_global
                            _mos_override_count += 1

        log(f'V69 MoS Apex: Applied Stage-2 Specialist overrides to {_mos_override_count} ambiguous sibling samples (conf >= 0.60)')
    except Exception as _e_mos:
        log(f'V69 MoS Warning: Exception in Stage-2 Specialist: {_e_mos}. Retaining Stage-1 V68 safe predictions.')
'''

with open('champion_v69/champion-v69.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update cell 0 title
nb['cells'][0]['source'] = ['# Name That LLM Champion V69 Mixture of Specialists Apex (Targeting 0.670+ Breakthrough)\n']

# Update cell 6 infer.py
src = ''.join(nb['cells'][6]['source'])
target = '''    log(f'V62 Anchor-Hungarian: locked {_n_anchors_locked} anchors, reassigned {_n_hungarian_reassigned} ambiguous items across {len(_dense_clusters)} dense clusters')'''

assert target in src, "Target string not found!"
src = src.replace(target, mos_code, 1)
nb['cells'][6]['source'] = [src]

with open('champion_v69/champion-v69.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated champion-v69.ipynb successfully!')

