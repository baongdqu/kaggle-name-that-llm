import json

nb = json.load(open('champion_v2/champion-v2.ipynb', encoding='utf-8'))
code = ''.join(nb['cells'][1]['source'])

openai_gate_code = """
# ------------------------------------------------ 1c) Multi-Model Gate (OpenAI Sub-model Resolver via External Corpus)
OPENAI_GATE_LAM = 1.2
OPENAI_MODELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra']
OPENAI_IDX = np.array([i for i, l in enumerate(LABELS) if l in OPENAI_MODELS])
DC_TE_OPENAI = None

_ext_corpus_hits = glob.glob('/kaggle/input/**/external_short_wide.parquet', recursive=True)
if _ext_corpus_hits:
    try:
        _ext_df = pd.read_parquet(_ext_corpus_hits[0])
        _ext_rows = []
        for col in OPENAI_MODELS:
            if col in _ext_df.columns:
                sub = _ext_df[['id', col]].dropna().rename(columns={col: 'text'})
                sub['model'] = col
                _ext_rows.append(sub)
        if _ext_rows:
            _ext_data = pd.concat(_ext_rows, ignore_index=True)
            _m_oai = np.isin(y, OPENAI_IDX)
            _ft_oai = [t for t, k in zip(TR, _m_oai) if k] + _ext_data['text'].astype(str).tolist()
            _fy_oai = np.concatenate([y[_m_oai], _ext_data['model'].map({l: i for i, l in enumerate(LABELS)}).values])
            
            A_oai, B_oai = [], []
            for analyzer, lo, hi, lower, maxf, w in SPEC:
                tf = shape_text if analyzer == 'shape' else function_text if analyzer == 'function' else str
                actual = 'char' if analyzer == 'shape' else 'word' if analyzer == 'function' else analyzer
                opts = dict(analyzer=actual, ngram_range=(lo, hi), lowercase=lower, sublinear_tf=True, min_df=2,
                            max_features=maxf, dtype=np.float64)
                if actual == 'word':
                    opts['token_pattern'] = r"(?u)\\b\\w+\\b|[^\\w\\s]"
                v = TfidfVectorizer(**opts)
                A_oai.append(v.fit_transform([tf(t) for t in _ft_oai]) * w)
                B_oai.append(v.transform([tf(t) for t in TE]) * w)
            _svc_oai = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(hstack(A_oai, format='csr'), _fy_oai)
            assert list(_svc_oai.classes_) == list(OPENAI_IDX)
            DC_TE_OPENAI = _svc_oai.decision_function(hstack(B_oai, format='csr'))
            print('OpenAI gate model trained on', len(_ft_oai), 'rows', flush=True)
            del A_oai, B_oai, _svc_oai, _ext_data, _ext_rows
    except Exception as e:
        print('Could not load OpenAI external corpus:', e, flush=True)
"""

target_gate_anchor = "print('claude gate model trained on', len(_ft), 'rows', flush=True)\n    del A_cg, B_cg, _svc, _api\nelse:\n    print('Claude API passages not found, continuing without gate', flush=True)"
assert target_gate_anchor in code, "Claude gate anchor not found"
code = code.replace(target_gate_anchor, target_gate_anchor + "\n" + openai_gate_code)

target_refine_anchor = "pred[_isc] = _new[_isc]"
assert target_refine_anchor in code, "refine anchor not found"

openai_refine_code = """pred[_isc] = _new[_isc]
if 'DC_TE_OPENAI' in globals() and DC_TE_OPENAI is not None:
    _iso = np.isin(pred, OPENAI_IDX)
    _so = np.log(np.clip(P[:, OPENAI_IDX], 1e-12, 1)) + OPENAI_GATE_LAM * DC_TE_OPENAI
    _new_oai = OPENAI_IDX[_so.argmax(1)]
    print('OpenAI gate refined', int((_new_oai[_iso] != pred[_iso]).sum()), 'of', int(_iso.sum()), 'OpenAI predictions', flush=True)
    pred[_iso] = _new_oai[_iso]"""

code = code.replace(target_refine_anchor, openai_refine_code)

nb['cells'][1]['source'] = code
json.dump(nb, open('champion_v2/champion-v2.ipynb', 'w', encoding='utf-8'), indent=1)
print("SUCCESS: Injected OpenAI Gate into champion-v2.ipynb!")
