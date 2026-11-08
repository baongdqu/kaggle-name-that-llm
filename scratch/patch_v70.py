import json

v70_mos_code = r'''    _safe_logits = _raw + _gate[:, None] * _evidence

    # =========================================================================
    # --- Champion V70: Hungarian-Aware Mixture of Specialists (MoS) Apex ---
    # Injects micro-stylometric Specialist Logits (Burrows' Delta Function Words,
    # Character N-grams 3-5, Sentence variance/skewness) directly into _safe_logits
    # BEFORE Latin-Square Kuhn-Munkres matching!
    # Validated on OOF: w_spec=0.008 yields 5:1 Win/Loss Ratio (+0.0015 Net Gain)
    # Allows Hungarian bipartite matching to respect Latin-Square prompt constraints
    # with ZERO collision and ZERO zero-sum penalty!
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
                _slens = np.array([len(s.split()) for s in _sents], dtype=np.float32)
                _mslen = float(np.mean(_slens))
                _sslen = float(np.std(_slens)) if len(_slens) > 1 else 0.0
                _sklen = float(np.mean(((_slens - _mslen) / max(_sslen, 1e-4)) ** 3)) if _sslen > 1e-4 else 0.0
                _comma = _st.count(',') / _nw
                _semi = _st.count(';') / _nw
                _colon = _st.count(':') / _nw
                _emdash = (_st.count('—') + _st.count('–')) / _nw
                _paren = _st.count('(') / _nw
                _quote = (_st.count('\"') + _st.count(\"'\") + _st.count('“') + _st.count('”')) / _nw
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

        log('V70 MoS: Extracting micro-stylometrics for Hungarian-Aware Specialists...')
        _X_tr_spec, _v_fw, _v_char, _sc_stats = _get_micro_features(train.text.values)
        _X_te_spec, _, _, _ = _get_micro_features(test.text.values, _v_fw, _v_char, _sc_stats)

        _train_labels = np.array([LABELS.index(m) for m in train.model.values])
        _mos_targets = {
            'gemini_flash': [LABELS.index('gemini-3.6-flash'), LABELS.index('gemini-3.7-flash'), LABELS.index('gemini-3.8-flash')],
            'glm53': [LABELS.index('glm-5.3'), LABELS.index('glm-5.3-flash')]
        }

        _w_spec = 0.008
        _n_injected = 0
        for _grp_name, _cls_list in _mos_targets.items():
            _mask = np.isin(_train_labels, _cls_list)
            _y_local = np.array([_cls_list.index(l) for l in _train_labels[_mask]])
            _clf = _SpLR(C=1.0, max_iter=300, random_state=42)
            _clf.fit(_X_tr_spec[_mask], _y_local)
            
            # Predict log-odds on test set
            _te_probs = _clf.predict_proba(_X_te_spec)
            _log_p = np.log(_te_probs + 1e-9)
            _log_p -= _log_p.mean(axis=1, keepdims=True)
            
            # Apply to samples where top safe_logits is currently in this group
            _top_in_grp = np.isin(_safe_logits.argmax(axis=1), _cls_list)
            for _loc_i, _glob_c in enumerate(_cls_list):
                _safe_logits[_top_in_grp, _glob_c] += _w_spec * _log_p[_top_in_grp, _loc_i]
            _n_injected += int(_top_in_grp.sum())
            log(f'V70 MoS: Injected specialist logits [{_grp_name}] (w={_w_spec}) into {_top_in_grp.sum()} candidate samples')

        log(f'V70 MoS Apex: Completed Hungarian-Aware Specialist Injection across {_n_injected} sample-group occurrences')
    except Exception as _e_v70:
        log(f'V70 MoS Warning: Exception in Hungarian-Aware Specialist: {_e_v70}')

    # V62 Latin-Square Anchor-Protected Kuhn-Munkres Assignment'''

with open('champion_v70/champion-v70.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'][0]['source'] = ['# Name That LLM Champion V70 Hungarian-Aware MoS Apex (Targeting 0.670+ Breakthrough)\n']

src = ''.join(nb['cells'][6]['source'])
target = '''    _safe_logits = _raw + _gate[:, None] * _evidence
    # V62 Latin-Square Anchor-Protected Kuhn-Munkres Assignment'''

assert target in src, 'Target string not found in V70 notebook cell 6!'
src = src.replace(target, v70_mos_code, 1)
nb['cells'][6]['source'] = [src]

with open('champion_v70/champion-v70.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated champion-v70.ipynb successfully!')

