import json

nb = json.load(open('champion_v1__kernel/champion-v1.ipynb', encoding='utf-8'))
code = ''.join(nb['cells'][1]['source'])

claude_gate_injection = """
# ------------------------------------------------ 1b) Multi-Model Gate (Claude Sub-model Resolver via External API Data)
CLAUDE_GATE_LAM = 1.5
_api_hits = glob.glob('/kaggle/input/**/ntl_extra_passages_claude.csv', recursive=True)
CLAUDE_IDX = np.array([i for i, l in enumerate(LABELS) if l.startswith('claude')])
DC_TE = None
if _api_hits:
    _api = pd.read_csv(_api_hits[0])
    _m = np.isin(y, CLAUDE_IDX)
    _ft = [t for t, k in zip(TR, _m) if k] + _api['text'].astype(str).tolist()
    _fy = np.concatenate([y[_m], _api['model'].map({l: i for i, l in enumerate(LABELS)}).values])
    A_cg, B_cg = [], []
    for analyzer, lo, hi, lower, maxf, w in SPEC:
        tf = shape_text if analyzer == 'shape' else function_text if analyzer == 'function' else str
        actual = 'char' if analyzer == 'shape' else 'word' if analyzer == 'function' else analyzer
        opts = dict(analyzer=actual, ngram_range=(lo, hi), lowercase=lower, sublinear_tf=True, min_df=2,
                    max_features=maxf, dtype=np.float64)
        if actual == 'word':
            opts['token_pattern'] = r"(?u)\\b\\w+\\b|[^\\w\\s]"
        v = TfidfVectorizer(**opts)
        A_cg.append(v.fit_transform([tf(t) for t in _ft]) * w)
        B_cg.append(v.transform([tf(t) for t in TE]) * w)
    _svc = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(hstack(A_cg, format='csr'), _fy)
    assert list(_svc.classes_) == list(CLAUDE_IDX)
    DC_TE = _svc.decision_function(hstack(B_cg, format='csr'))
    print('claude gate model trained on', len(_ft), 'rows', flush=True)
    del A_cg, B_cg, _svc, _api
else:
    print('Claude API passages not found, continuing without gate', flush=True)
"""

target_anchor = "add('stylegram', p)\ndel A, B, svc"
assert target_anchor in code, "stylegram anchor not found"
code = code.replace(target_anchor, target_anchor + "\n" + claude_gate_injection)

post_anchor = "pred = P.argmax(1)"
assert post_anchor in code, "pred anchor not found"

post_injection = """pred = P.argmax(1)
if 'DC_TE' in globals() and DC_TE is not None:
    _isc = np.isin(pred, CLAUDE_IDX)
    _s = np.log(np.clip(P[:, CLAUDE_IDX], 1e-12, 1)) + CLAUDE_GATE_LAM * DC_TE
    _new = CLAUDE_IDX[_s.argmax(1)]
    print('claude gate refined', int((_new[_isc] != pred[_isc]).sum()), 'of', int(_isc.sum()), 'Claude predictions', flush=True)
    pred[_isc] = _new[_isc]"""

code = code.replace(post_anchor, post_injection)

nb['cells'][1]['source'] = code
json.dump(nb, open('champion_v1__kernel/champion-v1.ipynb', 'w', encoding='utf-8'), indent=1)
print("SUCCESS: Injected Claude Gate & External Passages into champion-v1.ipynb!")
