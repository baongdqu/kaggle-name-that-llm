import json
import os
import ast

os.makedirs('champion_v13', exist_ok=True)

# 1. Metadata for V13
meta_v13 = {
  "id": "baonguyenduyquoc/name-that-llm-champion-v13-crossfamily-gate",
  "title": "Name That LLM Champion V13 CrossFamily Gate",
  "code_file": "champion-v13.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": True,
  "enable_gpu": True,
  "enable_tpu": False,
  "enable_internet": False,
  "dataset_sources": [
    "bbucxi/qwen35-9b-safetensors",
    "pjmathematician/ultrabert-380m-ntl",
    "seyitkaangunes/ntl-extra-api-passages",
    "olegzholobov/name-that-llm-response-corpus"
  ],
  "kernel_sources": [
    "lucasboesen/name-that-llm-glm4-9b-trunc25",
    "lucasboesen/name-that-llm-extract-encoders2",
    "lucasboesen/name-that-llm-electra-large-weights",
    "lucasboesen/name-that-llm-desklib-weights",
    "lightningv08/ntl-lora24-fivefold-training",
    "lucasboesen/name-that-llm-train-lora24-s7",
    "lucasboesen/name-that-llm-train-tokattn",
    "lucasboesen/name-that-llm-train-tokattn2"
  ],
  "competition_sources": [
    "name-that-llm"
  ],
  "model_sources": [
    "qwen-lm/qwen-3/Transformers/4b-base/1",
    "qwen-lm/qwen-3/Transformers/8b-base/1",
    "qwen-lm/qwen2.5/Transformers/7b/1"
  ],
  "machine_shape": "NvidiaTeslaT4"
}

with open('champion_v13/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta_v13, f, indent=2)

print("Saved champion_v13/kernel-metadata.json")

# 2. Extract base code from V12 (which already has Unified 4-Way Chinese Gate)
nb_v12 = json.load(open('champion_v12/champion-v12.ipynb', encoding='utf-8'))
code_v12 = nb_v12['cells'][1]['source']

# 3. Anchor 1: Training block
# In V12, we had GLM/Chinese gate training inside the external corpus block.
# Let's inspect where Chinese gate was defined in V12:
assert "CH_MODELS = ['glm-5.3', 'glm-5.3-flash', 'kimi-k3', 'qwen3.8-max']" in code_v12, "Chinese gate in V12 not found!"

# We want to add Universal 18-class mega-gate training right after the Chinese gate in the external corpus block!
target_ch_end = """            _svc_ch = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(hstack(A_ch, format='csr'), _fy_ch)
            assert list(_svc_ch.classes_) == list(CH_IDX)
            DC_TE_CH = _svc_ch.decision_function(hstack(B_ch, format='csr'))
            print('Unified Chinese frontier gate model trained on', len(_ft_ch), 'rows, DC shape:', DC_TE_CH.shape, flush=True)
            del A_ch, B_ch, _svc_ch, _ext_data_ch, _ext_rows_ch"""

assert target_ch_end in code_v12, "Target CH end not found in code_v12!"

v13_train_injection = target_ch_end + """

        # Universal 18-Class Cross-Family Mega-Gate (Trained on 9,000+ balanced external passages)
        GLOBAL_GATE_LAM = 0.30
        DC_TE_GLOBAL = None
        _rows_global = []
        for col in LABELS:
            if col in _ext_df.columns:
                sub = _ext_df[['id', col]].dropna().rename(columns={col: 'text'})
                sub['model'] = col
                if len(sub) > 450:
                    sub = sub.sample(450, random_state=42)
                _rows_global.append(sub)
        if _rows_global:
            _global_data = pd.concat(_rows_global, ignore_index=True)
            _ft_global = TR + _global_data['text'].astype(str).tolist()
            _fy_global = np.concatenate([y, _global_data['model'].map({l: i for i, l in enumerate(LABELS)}).values])
            
            A_glob, B_glob = [], []
            for analyzer, lo, hi, lower, maxf, w in SPEC:
                tf = shape_text if analyzer == 'shape' else function_text if analyzer == 'function' else str
                actual = 'char' if analyzer == 'shape' else 'word' if analyzer == 'function' else analyzer
                opts = dict(analyzer=actual, ngram_range=(lo, hi), lowercase=lower, sublinear_tf=True, min_df=2,
                            max_features=maxf, dtype=np.float64)
                if actual == 'word':
                    opts['token_pattern'] = r"(?u)\\b\\w+\\b|[^\\w\\s]"
                v = TfidfVectorizer(**opts)
                A_glob.append(v.fit_transform([tf(t) for t in _ft_global]) * w)
                B_glob.append(v.transform([tf(t) for t in TE]) * w)
            _svc_glob = LinearSVC(C=0.4, random_state=42, max_iter=3000).fit(hstack(A_glob, format='csr'), _fy_global)
            assert list(_svc_glob.classes_) == list(range(K))
            DC_TE_GLOBAL = _svc_glob.decision_function(hstack(B_glob, format='csr'))
            print('Universal 18-class cross-family mega-gate trained on', len(_ft_global), 'rows, DC shape:', DC_TE_GLOBAL.shape, flush=True)
            del A_glob, B_glob, _svc_glob, _global_data, _rows_global"""

# 4. Anchor 2: Inference / Refinement block
# In V12, the refinement code was:
v12_old_refine_anchor = """pred = P.argmax(1)
if 'DC_TE' in globals() and DC_TE is not None:
    _isc = np.isin(pred, CLAUDE_IDX)
    _s = np.log(np.clip(P[:, CLAUDE_IDX], 1e-12, 1)) + CLAUDE_GATE_LAM * DC_TE
    _new = CLAUDE_IDX[_s.argmax(1)]
    print('claude gate refined', int((_new[_isc] != pred[_isc]).sum()), 'of', int(_isc.sum()), 'Claude predictions', flush=True)
    pred[_isc] = _new[_isc]
if 'DC_TE_OPENAI' in globals() and DC_TE_OPENAI is not None:
    _iso = np.isin(pred, OPENAI_IDX)
    _so = np.log(np.clip(P[:, OPENAI_IDX], 1e-12, 1)) + OPENAI_GATE_LAM * DC_TE_OPENAI
    _new_oai = OPENAI_IDX[_so.argmax(1)]
    print('OpenAI gate refined', int((_new_oai[_iso] != pred[_iso]).sum()), 'of', int(_iso.sum()), 'OpenAI predictions', flush=True)
    pred[_iso] = _new_oai[_iso]

if 'DC_TE_GEMINI' in globals() and DC_TE_GEMINI is not None:
    _isg = np.isin(pred, GEMINI_IDX)
    _sg = np.log(np.clip(P[:, GEMINI_IDX], 1e-12, 1)) + GEMINI_GATE_LAM * DC_TE_GEMINI
    _new_gem = GEMINI_IDX[_sg.argmax(1)]
    print('Gemini gate refined', int((_new_gem[_isg] != pred[_isg]).sum()), 'of', int(_isg.sum()), 'Gemini predictions', flush=True)
    pred[_isg] = _new_gem[_isg]

if 'DC_TE_CH' in globals() and DC_TE_CH is not None:
    _isch = np.isin(pred, CH_IDX)
    _sch = np.log(np.clip(P[:, CH_IDX], 1e-12, 1)) + CH_GATE_LAM * DC_TE_CH
    _new_ch = CH_IDX[_sch.argmax(1)]
    print('Unified Chinese gate refined', int((_new_ch[_isch] != pred[_isch]).sum()), 'of', int(_isch.sum()), 'Chinese predictions', flush=True)
    pred[_isch] = _new_ch[_isch]"""

assert v12_old_refine_anchor in code_v12, "v12_old_refine_anchor not found in code_v12!"

# In V13, we replace with Cross-Family Soft Posterior Refinement:
v13_new_refine_code = """# ------------------------------------------------ V13: Cross-Family Soft Posterior Calibration
P_REFINED = np.log(np.clip(P, 1e-12, 1)).copy()

if 'DC_TE_GLOBAL' in globals() and DC_TE_GLOBAL is not None:
    P_REFINED += GLOBAL_GATE_LAM * DC_TE_GLOBAL
    print('Universal 18-class cross-family calibration applied (lam =', GLOBAL_GATE_LAM, ')', flush=True)

if 'DC_TE' in globals() and DC_TE is not None:
    CLAUDE_GATE_LAM_V13 = 0.8
    P_REFINED[:, CLAUDE_IDX] += CLAUDE_GATE_LAM_V13 * (DC_TE - DC_TE.mean(1, keepdims=True))
    print('Claude intra-family micro-gate applied (lam =', CLAUDE_GATE_LAM_V13, ')', flush=True)

if 'DC_TE_OPENAI' in globals() and DC_TE_OPENAI is not None:
    OPENAI_GATE_LAM_V13 = 0.8
    P_REFINED[:, OPENAI_IDX] += OPENAI_GATE_LAM_V13 * (DC_TE_OPENAI - DC_TE_OPENAI.mean(1, keepdims=True))
    print('OpenAI intra-family micro-gate applied (lam =', OPENAI_GATE_LAM_V13, ')', flush=True)

if 'DC_TE_GEMINI' in globals() and DC_TE_GEMINI is not None:
    GEMINI_GATE_LAM_V13 = 0.8
    P_REFINED[:, GEMINI_IDX] += GEMINI_GATE_LAM_V13 * (DC_TE_GEMINI - DC_TE_GEMINI.mean(1, keepdims=True))
    print('Gemini intra-family micro-gate applied (lam =', GEMINI_GATE_LAM_V13, ')', flush=True)

if 'DC_TE_CH' in globals() and DC_TE_CH is not None:
    CH_GATE_LAM_V13 = 0.8
    P_REFINED[:, CH_IDX] += CH_GATE_LAM_V13 * (DC_TE_CH - DC_TE_CH.mean(1, keepdims=True))
    print('Chinese intra-family micro-gate applied (lam =', CH_GATE_LAM_V13, ')', flush=True)

pred = P_REFINED.argmax(1)
print('Cross-family gating modified', int((pred != P.argmax(1)).sum()), 'of', len(P), 'predictions from base Sinkhorn', flush=True)"""

code_v13 = code_v12.replace(target_ch_end, v13_train_injection, 1)
code_v13 = code_v13.replace(v12_old_refine_anchor, v13_new_refine_code, 1)

# Validate syntax
ast.parse(code_v13)
print("SUCCESS: V13 syntax is 100% valid Python!")

# Write notebook
nb_v13 = json.load(open('champion_v4_quadgate/champion-v4-quadgate.ipynb', encoding='utf-8'))
nb_v13['cells'][0]['source'] = [
    '# Name That LLM - Champion Ensemble V13 (Cross-Family Soft Gating + 18-Class Universal Mega-Gate)\n',
    '\n',
    'Team Bao Ng 🔫🔫🔫 | Target Top 10 (>0.565) & Top 1 (0.616)\n'
]
nb_v13['cells'][1]['source'] = code_v13
with open('champion_v13/champion-v13.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_v13, f, indent=1)

print("Saved champion_v13/champion-v13.ipynb successfully!")
