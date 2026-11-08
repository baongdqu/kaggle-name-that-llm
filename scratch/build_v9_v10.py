import json
import os
import ast

os.makedirs('champion_v9', exist_ok=True)
os.makedirs('champion_v10', exist_ok=True)

# 1. Metadata for V9
meta_v9 = {
  "id": "baonguyenduyquoc/name-that-llm-champion-v9-pseudo-refit",
  "title": "Name That LLM - Champion V9 Transductive Pseudo-Refit",
  "code_file": "champion-v9.ipynb",
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

with open('champion_v9/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta_v9, f, indent=2)

# 2. Metadata for V10
meta_v10 = {
  "id": "baonguyenduyquoc/name-that-llm-champion-v10-balanced-refit",
  "title": "Name That LLM - Champion V10 Class-Balanced Pseudo-Refit",
  "code_file": "champion-v10.ipynb",
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

with open('champion_v10/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta_v10, f, indent=2)

# 3. Base code from V4
nb_v4 = json.load(open('champion_v4_quadgate/champion-v4-quadgate.ipynb', encoding='utf-8'))
code_base = ''.join(nb_v4['cells'][1]['source'])

# Check and update scipy.sparse imports
import_anchor = "from scipy.sparse import hstack"
import_replacement = "from scipy.sparse import hstack, vstack"
assert import_anchor in code_base, "Import anchor not found!"
code_base = code_base.replace(import_anchor, import_replacement, 1)

# Stylegram anchor: keep A_STYLE, B_STYLE and P_STYLE_ORIG
style_anchor = """svc = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(hstack(A, format='csr'), y)
s = svc.decision_function(hstack(B, format='csr'))
s = np.exp(s - s.max(1, keepdims=True))
p = np.zeros((len(TE), K))
p[:, svc.classes_] = s / s.sum(1, keepdims=True)
add('stylegram', p)
del A, B, svc"""

style_replacement = """A_STYLE = hstack(A, format='csr')
B_STYLE = hstack(B, format='csr')
svc = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(A_STYLE, y)
s = svc.decision_function(B_STYLE)
s = np.exp(s - s.max(1, keepdims=True))
p = np.zeros((len(TE), K))
p[:, svc.classes_] = s / s.sum(1, keepdims=True)
add('stylegram', p)
P_STYLE_ORIG = p.copy()
del A, B, svc"""

assert style_anchor in code_base, "Style anchor not found!"
code_base = code_base.replace(style_anchor, style_replacement, 1)

# Refinement end anchor: right after GLM gate refinement and before `sub = pd.DataFrame`
post_anchor = """if 'DC_TE_GLM' in globals() and DC_TE_GLM is not None:
    _isl = np.isin(pred, GLM_IDX)
    _sl = np.log(np.clip(P[:, GLM_IDX], 1e-12, 1)) + GLM_GATE_LAM * DC_TE_GLM
    _new_glm = GLM_IDX[_sl.argmax(1)]
    print('GLM gate refined', int((_new_glm[_isl] != pred[_isl]).sum()), 'of', int(_isl.sum()), 'GLM predictions', flush=True)
    pred[_isl] = _new_glm[_isl]"""

assert post_anchor in code_base, "Post anchor not found!"

# --- V9 Refit Block ---
v9_refit_code = """if 'DC_TE_GLM' in globals() and DC_TE_GLM is not None:
    _isl = np.isin(pred, GLM_IDX)
    _sl = np.log(np.clip(P[:, GLM_IDX], 1e-12, 1)) + GLM_GATE_LAM * DC_TE_GLM
    _new_glm = GLM_IDX[_sl.argmax(1)]
    print('GLM gate refined', int((_new_glm[_isl] != pred[_isl]).sum()), 'of', int(_isl.sum()), 'GLM predictions', flush=True)
    pred[_isl] = _new_glm[_isl]

# ------------------------------------------------ 4) Transductive Pseudo-Label Refit (V9 Global Margin)
if len(P) > 50 and 'A_STYLE' in globals() and 'B_STYLE' in globals():
    sorted_p = np.sort(P, axis=1)
    margins = sorted_p[:, -1] - sorted_p[:, -2]
    conf_thresh = 0.45
    mask_pseudo = margins >= conf_thresh
    max_pseudo = int(len(P) * 0.25)
    if mask_pseudo.sum() > max_pseudo:
        top_idx = np.argsort(margins)[::-1][:max_pseudo]
        mask_pseudo = np.zeros(len(P), dtype=bool)
        mask_pseudo[top_idx] = True
    pseudo_idx = np.where(mask_pseudo)[0]
    
    if len(pseudo_idx) >= 18 and len(np.unique(pred[pseudo_idx])) >= 3:
        print(f"Transductive Refit (V9): Found {len(pseudo_idx)} pseudo test samples across {len(np.unique(pred[pseudo_idx]))} classes (margin >= {conf_thresh:.2f})", flush=True)
        X_train_refit = vstack([A_STYLE, B_STYLE[pseudo_idx]], format='csr')
        y_train_refit = np.concatenate([y, pred[pseudo_idx]])
        svc_refit = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(X_train_refit, y_train_refit)
        s_refit = svc_refit.decision_function(B_STYLE)
        s_refit = np.exp(s_refit - s_refit.max(1, keepdims=True))
        p_refit = np.zeros((len(TE), K))
        p_refit[:, svc_refit.classes_] = s_refit / s_refit.sum(1, keepdims=True)
        
        LOGIT[:] -= W['stylegram'] * np.log(np.clip(P_STYLE_ORIG, 1e-9, 1))
        LOGIT[:] += W['stylegram'] * np.log(np.clip(p_refit, 1e-9, 1))
        print("Transductive Refit (V9): LOGIT updated with refitted stylegram", flush=True)
        
        P = np.exp(LOGIT - LOGIT.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        if BALANCE_ALPHA > 0 and len(P) >= 2:
            target = (1 - BALANCE_ALPHA) * P.sum(0) + BALANCE_ALPHA * np.full(K, len(P) / K)
            for _ in range(50):
                P *= (target / (P.sum(0) + 1e-12))[None, :]
                P /= P.sum(1, keepdims=True) + 1e-12
            print('Post-refit balancing changed', int((P.argmax(1) != LOGIT.argmax(1)).sum()), 'of', len(P), flush=True)
        pred = P.argmax(1)
        if 'DC_TE' in globals() and DC_TE is not None:
            _isc = np.isin(pred, CLAUDE_IDX)
            _s = np.log(np.clip(P[:, CLAUDE_IDX], 1e-12, 1)) + CLAUDE_GATE_LAM * DC_TE
            _new = CLAUDE_IDX[_s.argmax(1)]
            pred[_isc] = _new[_isc]
        if 'DC_TE_OPENAI' in globals() and DC_TE_OPENAI is not None:
            _iso = np.isin(pred, OPENAI_IDX)
            _so = np.log(np.clip(P[:, OPENAI_IDX], 1e-12, 1)) + OPENAI_GATE_LAM * DC_TE_OPENAI
            _new_oai = OPENAI_IDX[_so.argmax(1)]
            pred[_iso] = _new_oai[_iso]
        if 'DC_TE_GEMINI' in globals() and DC_TE_GEMINI is not None:
            _isg = np.isin(pred, GEMINI_IDX)
            _sg = np.log(np.clip(P[:, GEMINI_IDX], 1e-12, 1)) + GEMINI_GATE_LAM * DC_TE_GEMINI
            _new_gem = GEMINI_IDX[_sg.argmax(1)]
            pred[_isg] = _new_gem[_isg]
        if 'DC_TE_GLM' in globals() and DC_TE_GLM is not None:
            _isl = np.isin(pred, GLM_IDX)
            _sl = np.log(np.clip(P[:, GLM_IDX], 1e-12, 1)) + GLM_GATE_LAM * DC_TE_GLM
            _new_glm = GLM_IDX[_sl.argmax(1)]
            pred[_isl] = _new_glm[_isl]
        del X_train_refit, y_train_refit, svc_refit, s_refit, p_refit
    else:
        print("Transductive Refit (V9): Insufficient high-confidence pseudo samples, skipping refit.", flush=True)"""

# --- V10 Refit Block ---
v10_refit_code = """if 'DC_TE_GLM' in globals() and DC_TE_GLM is not None:
    _isl = np.isin(pred, GLM_IDX)
    _sl = np.log(np.clip(P[:, GLM_IDX], 1e-12, 1)) + GLM_GATE_LAM * DC_TE_GLM
    _new_glm = GLM_IDX[_sl.argmax(1)]
    print('GLM gate refined', int((_new_glm[_isl] != pred[_isl]).sum()), 'of', int(_isl.sum()), 'GLM predictions', flush=True)
    pred[_isl] = _new_glm[_isl]

# ------------------------------------------------ 4) Transductive Class-Balanced Pseudo-Label Refit (V10)
if len(P) > 50 and 'A_STYLE' in globals() and 'B_STYLE' in globals():
    sorted_p = np.sort(P, axis=1)
    margins = sorted_p[:, -1] - sorted_p[:, -2]
    selected_indices = []
    MAX_PER_CLASS = 25
    MIN_MARGIN = 0.35
    for c in range(K):
        c_mask = (pred == c) & (margins >= MIN_MARGIN)
        c_indices = np.where(c_mask)[0]
        if len(c_indices) > 0:
            c_sorted = c_indices[np.argsort(margins[c_indices])[::-1]]
            selected_indices.extend(c_sorted[:MAX_PER_CLASS].tolist())
    
    pseudo_idx = np.array(selected_indices, dtype=int)
    
    if len(pseudo_idx) >= 18 and len(np.unique(pred[pseudo_idx])) >= 3:
        print(f"Transductive Class-Balanced Refit (V10): Selected {len(pseudo_idx)} samples across {len(np.unique(pred[pseudo_idx]))} classes (max {MAX_PER_CLASS}/class, margin >= {MIN_MARGIN})", flush=True)
        X_train_refit = vstack([A_STYLE, B_STYLE[pseudo_idx]], format='csr')
        y_train_refit = np.concatenate([y, pred[pseudo_idx]])
        svc_refit = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(X_train_refit, y_train_refit)
        s_refit = svc_refit.decision_function(B_STYLE)
        s_refit = np.exp(s_refit - s_refit.max(1, keepdims=True))
        p_refit = np.zeros((len(TE), K))
        p_refit[:, svc_refit.classes_] = s_refit / s_refit.sum(1, keepdims=True)
        
        LOGIT[:] -= W['stylegram'] * np.log(np.clip(P_STYLE_ORIG, 1e-9, 1))
        LOGIT[:] += W['stylegram'] * np.log(np.clip(p_refit, 1e-9, 1))
        print("Transductive Class-Balanced Refit (V10): LOGIT updated with refitted stylegram", flush=True)
        
        P = np.exp(LOGIT - LOGIT.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        if BALANCE_ALPHA > 0 and len(P) >= 2:
            target = (1 - BALANCE_ALPHA) * P.sum(0) + BALANCE_ALPHA * np.full(K, len(P) / K)
            for _ in range(50):
                P *= (target / (P.sum(0) + 1e-12))[None, :]
                P /= P.sum(1, keepdims=True) + 1e-12
            print('Post-refit balancing changed', int((P.argmax(1) != LOGIT.argmax(1)).sum()), 'of', len(P), flush=True)
        pred = P.argmax(1)
        if 'DC_TE' in globals() and DC_TE is not None:
            _isc = np.isin(pred, CLAUDE_IDX)
            _s = np.log(np.clip(P[:, CLAUDE_IDX], 1e-12, 1)) + CLAUDE_GATE_LAM * DC_TE
            _new = CLAUDE_IDX[_s.argmax(1)]
            pred[_isc] = _new[_isc]
        if 'DC_TE_OPENAI' in globals() and DC_TE_OPENAI is not None:
            _iso = np.isin(pred, OPENAI_IDX)
            _so = np.log(np.clip(P[:, OPENAI_IDX], 1e-12, 1)) + OPENAI_GATE_LAM * DC_TE_OPENAI
            _new_oai = OPENAI_IDX[_so.argmax(1)]
            pred[_iso] = _new_oai[_iso]
        if 'DC_TE_GEMINI' in globals() and DC_TE_GEMINI is not None:
            _isg = np.isin(pred, GEMINI_IDX)
            _sg = np.log(np.clip(P[:, GEMINI_IDX], 1e-12, 1)) + GEMINI_GATE_LAM * DC_TE_GEMINI
            _new_gem = GEMINI_IDX[_sg.argmax(1)]
            pred[_isg] = _new_gem[_isg]
        if 'DC_TE_GLM' in globals() and DC_TE_GLM is not None:
            _isl = np.isin(pred, GLM_IDX)
            _sl = np.log(np.clip(P[:, GLM_IDX], 1e-12, 1)) + GLM_GATE_LAM * DC_TE_GLM
            _new_glm = GLM_IDX[_sl.argmax(1)]
            pred[_isl] = _new_glm[_isl]
        del X_train_refit, y_train_refit, svc_refit, s_refit, p_refit
    else:
        print("Transductive Class-Balanced Refit (V10): Insufficient high-confidence pseudo samples, skipping refit.", flush=True)"""

code_v9 = code_base.replace(post_anchor, v9_refit_code, 1)
code_v10 = code_base.replace(post_anchor, v10_refit_code, 1)

# Syntax verification
ast.parse(code_v9)
print("V9 code syntax valid!")
ast.parse(code_v10)
print("V10 code syntax valid!")

# Save notebooks
nb_v9 = json.load(open('champion_v4_quadgate/champion-v4-quadgate.ipynb', encoding='utf-8'))
nb_v9['cells'][1]['source'] = code_v9
with open('champion_v9/champion-v9.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_v9, f, indent=1)
print("Saved champion_v9/champion-v9.ipynb")

nb_v10 = json.load(open('champion_v4_quadgate/champion-v4-quadgate.ipynb', encoding='utf-8'))
nb_v10['cells'][1]['source'] = code_v10
with open('champion_v10/champion-v10.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_v10, f, indent=1)
print("Saved champion_v10/champion-v10.ipynb")
