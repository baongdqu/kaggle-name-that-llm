import json
import re

# Load V4 base notebook
base_nb_path = 'champion_v4_quadgate/champion-v4-quadgate.ipynb'
with open(base_nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

base_code = ''.join(nb['cells'][1]['source'])

# Helper function definition to insert into code
zscore_helper = """
def zscore_normalize(mat):
    if mat is None:
        return None
    mean = mat.mean(axis=1, keepdims=True)
    std = mat.std(axis=1, keepdims=True) + 1e-6
    return (mat - mean) / std
"""

# Let's inspect where the refinement and Sinkhorn happen in base_code
sinkhorn_anchor = "P = np.exp(LOGIT - LOGIT.max(1, keepdims=True))"
assert sinkhorn_anchor in base_code, "sinkhorn_anchor not found!"

# ==========================================
# 1. BUILD V7: ZScore Normalized QuadGate + Conservative Alpha (0.55)
# ==========================================
v7_code = base_code

# Replace BALANCE_ALPHA = 0.5 with BALANCE_ALPHA = 0.55
v7_code = v7_code.replace("BALANCE_ALPHA = 0.5\n", "BALANCE_ALPHA = 0.55\n")

# In V7, normalize DC before applying in post-hoc refinement
v7_refine_anchor = """if 'DC_TE' in globals() and DC_TE is not None:
    _isc = np.isin(pred, CLAUDE_IDX)
    _s = np.log(np.clip(P[:, CLAUDE_IDX], 1e-12, 1)) + CLAUDE_GATE_LAM * DC_TE
    _new = CLAUDE_IDX[_s.argmax(1)]
    print('claude gate refined', int((_new[_isc] != pred[_isc]).sum()), 'of', int(_isc.sum()), 'Claude predictions', flush=True)
    pred[_isc] = _new[_isc]"""

v7_refine_replacement = """
def zscore_normalize(mat):
    if mat is None: return None
    mean = mat.mean(axis=1, keepdims=True)
    std = mat.std(axis=1, keepdims=True) + 1e-6
    return (mat - mean) / std

if 'DC_TE' in globals() and DC_TE is not None:
    _isc = np.isin(pred, CLAUDE_IDX)
    _dc_c_norm = zscore_normalize(DC_TE)
    _s = np.log(np.clip(P[:, CLAUDE_IDX], 1e-12, 1)) + CLAUDE_GATE_LAM * _dc_c_norm
    _new = CLAUDE_IDX[_s.argmax(1)]
    print('claude gate refined', int((_new[_isc] != pred[_isc]).sum()), 'of', int(_isc.sum()), 'Claude predictions', flush=True)
    pred[_isc] = _new[_isc]"""

assert v7_refine_anchor in v7_code, "v7_refine_anchor not found!"
v7_code = v7_code.replace(v7_refine_anchor, v7_refine_replacement)

# Also normalize OpenAI, Gemini, GLM in V7
v7_code = v7_code.replace("OPENAI_GATE_LAM * DC_TE_OPENAI", "OPENAI_GATE_LAM * zscore_normalize(DC_TE_OPENAI)")
v7_code = v7_code.replace("GEMINI_GATE_LAM * DC_TE_GEMINI", "GEMINI_GATE_LAM * zscore_normalize(DC_TE_GEMINI)")
v7_code = v7_code.replace("GLM_GATE_LAM * DC_TE_GLM", "GLM_GATE_LAM * zscore_normalize(DC_TE_GLM)")

nb_v7 = json.loads(json.dumps(nb))
nb_v7['cells'][1]['source'] = v7_code
with open('champion_v7/champion-v7.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_v7, f, indent=1)
print("V7 Notebook built successfully!")


# ==========================================
# 2. BUILD V8: Pre-Sinkhorn QuadGate Injection (Gate-Integrated Marginal Balancing)
# ==========================================
v8_code = base_code

# In V8: Add ZScore normalized QuadGate directly into LOGIT before Softmax and Sinkhorn!
v8_presinkhorn_block = """
def zscore_normalize(mat):
    if mat is None: return None
    mean = mat.mean(axis=1, keepdims=True)
    std = mat.std(axis=1, keepdims=True) + 1e-6
    return (mat - mean) / std

# Pre-Sinkhorn Soft Logit Injection
if 'DC_TE' in globals() and DC_TE is not None:
    LOGIT[:, CLAUDE_IDX] += 0.8 * zscore_normalize(DC_TE)
    print('Injected calibrated Claude Gate into LOGIT prior to Sinkhorn', flush=True)
if 'DC_TE_OPENAI' in globals() and DC_TE_OPENAI is not None:
    LOGIT[:, OPENAI_IDX] += 0.8 * zscore_normalize(DC_TE_OPENAI)
    print('Injected calibrated OpenAI Gate into LOGIT prior to Sinkhorn', flush=True)
if 'DC_TE_GEMINI' in globals() and DC_TE_GEMINI is not None:
    LOGIT[:, GEMINI_IDX] += 0.7 * zscore_normalize(DC_TE_GEMINI)
    print('Injected calibrated Gemini Gate into LOGIT prior to Sinkhorn', flush=True)
if 'DC_TE_GLM' in globals() and DC_TE_GLM is not None:
    LOGIT[:, GLM_IDX] += 0.7 * zscore_normalize(DC_TE_GLM)
    print('Injected calibrated GLM Gate into LOGIT prior to Sinkhorn', flush=True)

P = np.exp(LOGIT - LOGIT.max(1, keepdims=True))"""

assert sinkhorn_anchor in v8_code
v8_code = v8_code.replace(sinkhorn_anchor, v8_presinkhorn_block)

# Since logits are already injected prior to Sinkhorn, post-hoc refinement is replaced by pure global Sinkhorn argmax
v8_old_posthoc = v7_refine_anchor + """
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

if 'DC_TE_GLM' in globals() and DC_TE_GLM is not None:
    _isl = np.isin(pred, GLM_IDX)
    _sl = np.log(np.clip(P[:, GLM_IDX], 1e-12, 1)) + GLM_GATE_LAM * DC_TE_GLM
    _new_glm = GLM_IDX[_sl.argmax(1)]
    print('GLM gate refined', int((_new_glm[_isl] != pred[_isl]).sum()), 'of', int(_isl.sum()), 'GLM predictions', flush=True)
    pred[_isl] = _new_glm[_isl]"""

v8_new_posthoc = """# Pre-Sinkhorn integration already balanced the gates with global optimal transport
print('Gate-integrated Sinkhorn prediction complete. Final predictions count:', len(pred), flush=True)"""

assert v8_old_posthoc in v8_code, "v8_old_posthoc not found in v8_code!"
v8_code = v8_code.replace(v8_old_posthoc, v8_new_posthoc)

nb_v8 = json.loads(json.dumps(nb))
nb_v8['cells'][1]['source'] = v8_code
with open('champion_v8/champion-v8.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_v8, f, indent=1)
print("V8 Notebook built successfully!")


# ==========================================
# 3. BUILD V9: Pre-Sinkhorn + Temperature Annealing (0.85) + Alpha (0.60)
# ==========================================
v9_code = v8_code

# In V9: Change BALANCE_ALPHA to 0.60
v9_code = v9_code.replace("BALANCE_ALPHA = 0.5\n", "BALANCE_ALPHA = 0.60\n")

# Add temperature scaling
v9_temp_anchor = "P = np.exp(LOGIT - LOGIT.max(1, keepdims=True))"
v9_temp_block = """TEMPERATURE = 0.85
P = np.exp((LOGIT - LOGIT.max(1, keepdims=True)) / TEMPERATURE)"""

assert v9_temp_anchor in v9_code
v9_code = v9_code.replace(v9_temp_anchor, v9_temp_block)

nb_v9 = json.loads(json.dumps(nb))
nb_v9['cells'][1]['source'] = v9_code
with open('champion_v9/champion-v9.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_v9, f, indent=1)
print("V9 Notebook built successfully!")
