import json
import os
import ast

os.makedirs('champion_v11', exist_ok=True)
os.makedirs('champion_v12', exist_ok=True)

# 1. Metadata for V11
meta_v11 = {
  "id": "baonguyenduyquoc/name-that-llm-champion-v11-hexagate",
  "title": "Name That LLM Champion V11 HexaGate",
  "code_file": "champion-v11.ipynb",
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

with open('champion_v11/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta_v11, f, indent=2)

# 2. Metadata for V12
meta_v12 = {
  "id": "baonguyenduyquoc/name-that-llm-champion-v12-chinese-gate",
  "title": "Name That LLM Champion V12 Chinese Gate",
  "code_file": "champion-v12.ipynb",
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

with open('champion_v12/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta_v12, f, indent=2)

# 3. Load V4 Base Notebook (The 0.552 record holder)
nb_v4 = json.load(open('champion_v4_quadgate/champion-v4-quadgate.ipynb', encoding='utf-8'))
code_v4 = ''.join(nb_v4['cells'][1]['source'])

# Anchor for GLM Gate Training end:
glm_train_anchor = """            _svc_glm = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(hstack(A_glm, format='csr'), _fy_glm)
            assert list(_svc_glm.classes_) == list(GLM_IDX)
            _dc_glm_raw = _svc_glm.decision_function(hstack(B_glm, format='csr'))
            # Convert binary 1D decision function to 2D column stack [class0, class1]
            DC_TE_GLM = np.column_stack([-_dc_glm_raw, _dc_glm_raw])
            print('GLM gate model trained on', len(_ft_glm), 'rows, DC shape:', DC_TE_GLM.shape, flush=True)
            del A_glm, B_glm, _svc_glm, _ext_data_glm, _ext_rows_glm"""

assert glm_train_anchor in code_v4, "GLM train anchor not found!"

# Anchor for full GLM Gate block (used in V12 replacement):
full_glm_train_anchor = """        # GLM Gate Training (2 ZAI GLM models)
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
            del A_glm, B_glm, _svc_glm, _ext_data_glm, _ext_rows_glm"""

assert full_glm_train_anchor in code_v4, "Full GLM train anchor not found!"

# Anchor for GLM refine at end:
glm_refine_anchor = """if 'DC_TE_GLM' in globals() and DC_TE_GLM is not None:
    _isl = np.isin(pred, GLM_IDX)
    _sl = np.log(np.clip(P[:, GLM_IDX], 1e-12, 1)) + GLM_GATE_LAM * DC_TE_GLM
    _new_glm = GLM_IDX[_sl.argmax(1)]
    print('GLM gate refined', int((_new_glm[_isl] != pred[_isl]).sum()), 'of', int(_isl.sum()), 'GLM predictions', flush=True)
    pred[_isl] = _new_glm[_isl]"""

assert glm_refine_anchor in code_v4, "GLM refine anchor not found!"

# ==============================================================================
# 4. BUILD V11 CODE (HexaGate: QuadGate + Kimi-Qwen Gate -> 100% 18 models)
# ==============================================================================
kq_train_code = """            _svc_glm = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(hstack(A_glm, format='csr'), _fy_glm)
            assert list(_svc_glm.classes_) == list(GLM_IDX)
            _dc_glm_raw = _svc_glm.decision_function(hstack(B_glm, format='csr'))
            # Convert binary 1D decision function to 2D column stack [class0, class1]
            DC_TE_GLM = np.column_stack([-_dc_glm_raw, _dc_glm_raw])
            print('GLM gate model trained on', len(_ft_glm), 'rows, DC shape:', DC_TE_GLM.shape, flush=True)
            del A_glm, B_glm, _svc_glm, _ext_data_glm, _ext_rows_glm

        # Kimi-Qwen Gate Training (2 Chinese Frontier models: kimi-k3 vs qwen3.8-max)
        KQ_GATE_LAM = 1.0
        KQ_MODELS = ['kimi-k3', 'qwen3.8-max']
        KQ_IDX = np.array([i for i, l in enumerate(LABELS) if l in KQ_MODELS])
        DC_TE_KQ = None
        
        _ext_rows_kq = []
        for col in KQ_MODELS:
            if col in _ext_df.columns:
                sub = _ext_df[['id', col]].dropna().rename(columns={col: 'text'})
                sub['model'] = col
                _ext_rows_kq.append(sub)
        if _ext_rows_kq:
            _ext_data_kq = pd.concat(_ext_rows_kq, ignore_index=True)
            _m_kq = np.isin(y, KQ_IDX)
            _ft_kq = [t for t, k in zip(TR, _m_kq) if k] + _ext_data_kq['text'].astype(str).tolist()
            _fy_kq = np.concatenate([y[_m_kq], _ext_data_kq['model'].map({l: i for i, l in enumerate(LABELS)}).values])
            
            A_kq, B_kq = [], []
            for analyzer, lo, hi, lower, maxf, w in SPEC:
                tf = shape_text if analyzer == 'shape' else function_text if analyzer == 'function' else str
                actual = 'char' if analyzer == 'shape' else 'word' if analyzer == 'function' else analyzer
                opts = dict(analyzer=actual, ngram_range=(lo, hi), lowercase=lower, sublinear_tf=True, min_df=2,
                            max_features=maxf, dtype=np.float64)
                if actual == 'word':
                    opts['token_pattern'] = r"(?u)\\b\\w+\\b|[^\\w\\s]"
                v = TfidfVectorizer(**opts)
                A_kq.append(v.fit_transform([tf(t) for t in _ft_kq]) * w)
                B_kq.append(v.transform([tf(t) for t in TE]) * w)
            _svc_kq = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(hstack(A_kq, format='csr'), _fy_kq)
            assert list(_svc_kq.classes_) == list(KQ_IDX)
            _dc_kq_raw = _svc_kq.decision_function(hstack(B_kq, format='csr'))
            DC_TE_KQ = np.column_stack([-_dc_kq_raw, _dc_kq_raw])
            print('Kimi-Qwen gate model trained on', len(_ft_kq), 'rows, DC shape:', DC_TE_KQ.shape, flush=True)
            del A_kq, B_kq, _svc_kq, _ext_data_kq, _ext_rows_kq"""

kq_refine_code = """if 'DC_TE_GLM' in globals() and DC_TE_GLM is not None:
    _isl = np.isin(pred, GLM_IDX)
    _sl = np.log(np.clip(P[:, GLM_IDX], 1e-12, 1)) + GLM_GATE_LAM * DC_TE_GLM
    _new_glm = GLM_IDX[_sl.argmax(1)]
    print('GLM gate refined', int((_new_glm[_isl] != pred[_isl]).sum()), 'of', int(_isl.sum()), 'GLM predictions', flush=True)
    pred[_isl] = _new_glm[_isl]

if 'DC_TE_KQ' in globals() and DC_TE_KQ is not None:
    _iskq = np.isin(pred, KQ_IDX)
    _skq = np.log(np.clip(P[:, KQ_IDX], 1e-12, 1)) + KQ_GATE_LAM * DC_TE_KQ
    _new_kq = KQ_IDX[_skq.argmax(1)]
    print('Kimi-Qwen gate refined', int((_new_kq[_iskq] != pred[_iskq]).sum()), 'of', int(_iskq.sum()), 'Kimi-Qwen predictions', flush=True)
    pred[_iskq] = _new_kq[_iskq]"""

code_v11 = code_v4.replace(glm_train_anchor, kq_train_code, 1)
code_v11 = code_v11.replace(glm_refine_anchor, kq_refine_code, 1)

# ==============================================================================
# 5. BUILD V12 CODE (Unified 4-Way Chinese Frontier Gate: GLM + Kimi + Qwen-Max)
# ==============================================================================
ch4_train_code = """        # Unified Chinese Frontier Gate Training (4 models: glm-5.3, glm-5.3-flash, kimi-k3, qwen3.8-max)
        CH_GATE_LAM = 1.0
        CH_MODELS = ['glm-5.3', 'glm-5.3-flash', 'kimi-k3', 'qwen3.8-max']
        CH_IDX = np.array([i for i, l in enumerate(LABELS) if l in CH_MODELS])
        DC_TE_CH = None
        
        _ext_rows_ch = []
        for col in CH_MODELS:
            if col in _ext_df.columns:
                sub = _ext_df[['id', col]].dropna().rename(columns={col: 'text'})
                sub['model'] = col
                _ext_rows_ch.append(sub)
        if _ext_rows_ch:
            _ext_data_ch = pd.concat(_ext_rows_ch, ignore_index=True)
            _m_ch = np.isin(y, CH_IDX)
            _ft_ch = [t for t, k in zip(TR, _m_ch) if k] + _ext_data_ch['text'].astype(str).tolist()
            _fy_ch = np.concatenate([y[_m_ch], _ext_data_ch['model'].map({l: i for i, l in enumerate(LABELS)}).values])
            
            A_ch, B_ch = [], []
            for analyzer, lo, hi, lower, maxf, w in SPEC:
                tf = shape_text if analyzer == 'shape' else function_text if analyzer == 'function' else str
                actual = 'char' if analyzer == 'shape' else 'word' if analyzer == 'function' else analyzer
                opts = dict(analyzer=actual, ngram_range=(lo, hi), lowercase=lower, sublinear_tf=True, min_df=2,
                            max_features=maxf, dtype=np.float64)
                if actual == 'word':
                    opts['token_pattern'] = r"(?u)\\b\\w+\\b|[^\\w\\s]"
                v = TfidfVectorizer(**opts)
                A_ch.append(v.fit_transform([tf(t) for t in _ft_ch]) * w)
                B_ch.append(v.transform([tf(t) for t in TE]) * w)
            _svc_ch = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(hstack(A_ch, format='csr'), _fy_ch)
            assert list(_svc_ch.classes_) == list(CH_IDX)
            DC_TE_CH = _svc_ch.decision_function(hstack(B_ch, format='csr'))
            print('Unified Chinese frontier gate model trained on', len(_ft_ch), 'rows, DC shape:', DC_TE_CH.shape, flush=True)
            del A_ch, B_ch, _svc_ch, _ext_data_ch, _ext_rows_ch"""

ch4_refine_code = """if 'DC_TE_CH' in globals() and DC_TE_CH is not None:
    _isch = np.isin(pred, CH_IDX)
    _sch = np.log(np.clip(P[:, CH_IDX], 1e-12, 1)) + CH_GATE_LAM * DC_TE_CH
    _new_ch = CH_IDX[_sch.argmax(1)]
    print('Unified Chinese gate refined', int((_new_ch[_isch] != pred[_isch]).sum()), 'of', int(_isch.sum()), 'Chinese predictions', flush=True)
    pred[_isch] = _new_ch[_isch]"""

code_v12 = code_v4.replace(full_glm_train_anchor, ch4_train_code, 1)
code_v12 = code_v12.replace(glm_refine_anchor, ch4_refine_code, 1)

# 6. Syntax Validation
ast.parse(code_v11)
print("SUCCESS: V11 syntax is 100% valid!")
ast.parse(code_v12)
print("SUCCESS: V12 syntax is 100% valid!")

# 7. Write Notebooks
nb_v11 = json.load(open('champion_v4_quadgate/champion-v4-quadgate.ipynb', encoding='utf-8'))
nb_v11['cells'][0]['source'] = [
    '# Name That LLM - Champion Ensemble V11 (HexaGate: 100% Complete 18-Model Coverage)\n',
    '\n',
    'Team Bao Ng 🔫🔫🔫 | Target Top 1 (0.616)\n'
]
nb_v11['cells'][1]['source'] = code_v11
with open('champion_v11/champion-v11.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_v11, f, indent=1)
print("Saved champion_v11/champion-v11.ipynb")

nb_v12 = json.load(open('champion_v4_quadgate/champion-v4-quadgate.ipynb', encoding='utf-8'))
nb_v12['cells'][0]['source'] = [
    '# Name That LLM - Champion Ensemble V12 (Unified 4-Way Chinese Frontier Gate)\n',
    '\n',
    'Team Bao Ng 🔫🔫🔫 | Target Top 1 (0.616)\n'
]
nb_v12['cells'][1]['source'] = code_v12
with open('champion_v12/champion-v12.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_v12, f, indent=1)
print("Saved champion_v12/champion-v12.ipynb")
