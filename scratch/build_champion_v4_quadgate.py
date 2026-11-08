import json
import os
import shutil

os.makedirs('champion_v4_quadgate', exist_ok=True)

# 1. Write kernel metadata
meta = {
  "id": "baonguyenduyquoc/name-that-llm-champion-v4-quadgate",
  "title": "Name That LLM - Champion V4 QuadGate Soft Injection",
  "code_file": "champion-v4-quadgate.ipynb",
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

with open('champion_v4_quadgate/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)

# 2. Load V2 notebook as base
nb = json.load(open('champion_v2/champion-v2.ipynb', encoding='utf-8'))
code = ''.join(nb['cells'][1]['source'])

# 3. Add Gemini and GLM training code right after OpenAI gate
openai_gate_anchor = """            _svc_oai = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(hstack(A_oai, format='csr'), _fy_oai)
            assert list(_svc_oai.classes_) == list(OPENAI_IDX)
            DC_TE_OPENAI = _svc_oai.decision_function(hstack(B_oai, format='csr'))
            print('OpenAI gate model trained on', len(_ft_oai), 'rows', flush=True)
            del A_oai, B_oai, _svc_oai, _ext_data, _ext_rows
    except Exception as e:
        print('Could not load OpenAI external corpus:', e, flush=True)"""

gemini_and_glm_gate_code = """            _svc_oai = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(hstack(A_oai, format='csr'), _fy_oai)
            assert list(_svc_oai.classes_) == list(OPENAI_IDX)
            DC_TE_OPENAI = _svc_oai.decision_function(hstack(B_oai, format='csr'))
            print('OpenAI gate model trained on', len(_ft_oai), 'rows', flush=True)
            del A_oai, B_oai, _svc_oai, _ext_data, _ext_rows

        # Gemini Gate Training (4 Google Gemini models)
        GEMINI_GATE_LAM = 1.1
        GEMINI_MODELS = ['gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash']
        GEMINI_IDX = np.array([i for i, l in enumerate(LABELS) if l in GEMINI_MODELS])
        DC_TE_GEMINI = None
        
        _ext_rows_gemini = []
        for col in GEMINI_MODELS:
            if col in _ext_df.columns:
                sub = _ext_df[['id', col]].dropna().rename(columns={col: 'text'})
                sub['model'] = col
                _ext_rows_gemini.append(sub)
        if _ext_rows_gemini:
            _ext_data_gem = pd.concat(_ext_rows_gemini, ignore_index=True)
            _m_gem = np.isin(y, GEMINI_IDX)
            _ft_gem = [t for t, k in zip(TR, _m_gem) if k] + _ext_data_gem['text'].astype(str).tolist()
            _fy_gem = np.concatenate([y[_m_gem], _ext_data_gem['model'].map({l: i for i, l in enumerate(LABELS)}).values])
            
            A_gem, B_gem = [], []
            for analyzer, lo, hi, lower, maxf, w in SPEC:
                tf = shape_text if analyzer == 'shape' else function_text if analyzer == 'function' else str
                actual = 'char' if analyzer == 'shape' else 'word' if analyzer == 'function' else analyzer
                opts = dict(analyzer=actual, ngram_range=(lo, hi), lowercase=lower, sublinear_tf=True, min_df=2,
                            max_features=maxf, dtype=np.float64)
                if actual == 'word':
                    opts['token_pattern'] = r"(?u)\\b\\w+\\b|[^\\w\\s]"
                v = TfidfVectorizer(**opts)
                A_gem.append(v.fit_transform([tf(t) for t in _ft_gem]) * w)
                B_gem.append(v.transform([tf(t) for t in TE]) * w)
            _svc_gem = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(hstack(A_gem, format='csr'), _fy_gem)
            assert list(_svc_gem.classes_) == list(GEMINI_IDX)
            DC_TE_GEMINI = _svc_gem.decision_function(hstack(B_gem, format='csr'))
            print('Gemini gate model trained on', len(_ft_gem), 'rows', flush=True)
            del A_gem, B_gem, _svc_gem, _ext_data_gem, _ext_rows_gemini

        # GLM Gate Training (2 ZAI GLM models)
        GLM_GATE_LAM = 1.0
        GLM_MODELS = ['glm-5.3', 'glm-5.3-flash']
        GLM_IDX = np.array([i for i, l in enumerate(LABELS) if l in GLM_MODELS])
        DC_TE_GLM = None
        
        _ext_rows_glm = []
        for col in GLM_MODELS:
            if col in _ext_df.columns:
                sub = _ext_df[['id', col]].dropna().rename(columns={col: 'text'})
                sub['model'] = col
                _ext_rows_glm.append(sub)
        if _ext_rows_glm:
            _ext_data_glm = pd.concat(_ext_rows_glm, ignore_index=True)
            _m_glm = np.isin(y, GLM_IDX)
            _ft_glm = [t for t, k in zip(TR, _m_glm) if k] + _ext_data_glm['text'].astype(str).tolist()
            _fy_glm = np.concatenate([y[_m_glm], _ext_data_glm['model'].map({l: i for i, l in enumerate(LABELS)}).values])
            
            A_glm, B_glm = [], []
            for analyzer, lo, hi, lower, maxf, w in SPEC:
                tf = shape_text if analyzer == 'shape' else function_text if analyzer == 'function' else str
                actual = 'char' if analyzer == 'shape' else 'word' if analyzer == 'function' else analyzer
                opts = dict(analyzer=actual, ngram_range=(lo, hi), lowercase=lower, sublinear_tf=True, min_df=2,
                            max_features=maxf, dtype=np.float64)
                if actual == 'word':
                    opts['token_pattern'] = r"(?u)\\b\\w+\\b|[^\\w\\s]"
                v = TfidfVectorizer(**opts)
                A_glm.append(v.fit_transform([tf(t) for t in _ft_glm]) * w)
                B_glm.append(v.transform([tf(t) for t in TE]) * w)
            _svc_glm = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(hstack(A_glm, format='csr'), _fy_glm)
            assert list(_svc_glm.classes_) == list(GLM_IDX)
            _dc_glm_raw = _svc_glm.decision_function(hstack(B_glm, format='csr'))
            # Convert binary 1D decision function to 2D column stack [class0, class1]
            DC_TE_GLM = np.column_stack([-_dc_glm_raw, _dc_glm_raw])
            print('GLM gate model trained on', len(_ft_glm), 'rows, DC shape:', DC_TE_GLM.shape, flush=True)
            del A_glm, B_glm, _svc_glm, _ext_data_glm, _ext_rows_glm
    except Exception as e:
        print('Could not load QuadGate external corpus:', e, flush=True)"""

assert openai_gate_anchor in code, "openai_gate_anchor not found in code!"
code = code.replace(openai_gate_anchor, gemini_and_glm_gate_code)

# 4. Add Gemini and GLM refinement right after OpenAI refinement
openai_refine_anchor = """if 'DC_TE_OPENAI' in globals() and DC_TE_OPENAI is not None:
    _iso = np.isin(pred, OPENAI_IDX)
    _so = np.log(np.clip(P[:, OPENAI_IDX], 1e-12, 1)) + OPENAI_GATE_LAM * DC_TE_OPENAI
    _new_oai = OPENAI_IDX[_so.argmax(1)]
    print('OpenAI gate refined', int((_new_oai[_iso] != pred[_iso]).sum()), 'of', int(_iso.sum()), 'OpenAI predictions', flush=True)
    pred[_iso] = _new_oai[_iso]"""

quad_refine_code = """if 'DC_TE_OPENAI' in globals() and DC_TE_OPENAI is not None:
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

if 'DC_TE_GLM' in globals() and DC_TE_GLM is not None:
    _isl = np.isin(pred, GLM_IDX)
    _sl = np.log(np.clip(P[:, GLM_IDX], 1e-12, 1)) + GLM_GATE_LAM * DC_TE_GLM
    _new_glm = GLM_IDX[_sl.argmax(1)]
    print('GLM gate refined', int((_new_glm[_isl] != pred[_isl]).sum()), 'of', int(_isl.sum()), 'GLM predictions', flush=True)
    pred[_isl] = _new_glm[_isl]"""

assert openai_refine_anchor in code, "openai_refine_anchor not found in code!"
code = code.replace(openai_refine_anchor, quad_refine_code)

nb['cells'][1]['source'] = code
with open('champion_v4_quadgate/champion-v4-quadgate.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("SUCCESS: champion-v4-quadgate.ipynb created successfully!")
