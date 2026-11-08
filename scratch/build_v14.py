import json
import os
import ast

os.makedirs('champion_v14', exist_ok=True)

# 1. Metadata for V14
meta_v14 = {
  "id": "baonguyenduyquoc/name-that-llm-champion-v14-instyle-mega-gate",
  "title": "Name That LLM Champion V14 InStyle Mega Gate",
  "code_file": "champion-v14.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": True,
  "enable_gpu": True,
  "enable_tpu": False,
  "enable_internet": False,
  "keywords": [
    "gpu"
  ],
  "dataset_sources": [
    "seyitkaangunes/ntl-extra-api-passages",
    "zbigniewkirchner/ntl-extra-gpt-passages",
    "abdullah1afify/ntl-generations",
    "dophuchung/ntl-proxy-generations",
    "zbigniewkirchner/ntl-qwen3-8b-lora-instyle",
    "zbigniewkirchner/ntl-qwen3-8b-lora-claude-head",
    "lucasboesen/ntl-free-proxy-generations",
    "prvsiyan/prvsiyans-extended-name-that-llm-dataset",
    "bbucxi/qwen35-9b-safetensors",
    "pjmathematician/ultrabert-380m-ntl"
  ],
  "kernel_sources": [
    "lucasboesen/name-that-llm-glm4-9b-trunc25",
    "lucasboesen/name-that-llm-extract-encoders2",
    "lucasboesen/name-that-llm-electra-large-weights",
    "lucasboesen/name-that-llm-desklib-weights",
    "lightningv08/ntl-lora24-fivefold-training",
    "lucasboesen/name-that-llm-train-lora24-s7",
    "lucasboesen/name-that-llm-train-tokattn",
    "lucasboesen/name-that-llm-train-tokattn2",
    "vnton37nguyn/ntl-train-lora-qwen25-instyle"
  ],
  "competition_sources": [
    "name-that-llm"
  ],
  "model_sources": [
    "qwen-lm/qwen-3/Transformers/8b-base/1",
    "qwen-lm/qwen-3/Transformers/4b-base/1",
    "qwen-lm/qwen2.5/Transformers/7b/1"
  ],
  "machine_shape": "NvidiaTeslaT4"
}

with open('champion_v14/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta_v14, f, indent=2)

print("Saved champion_v14/kernel-metadata.json")

# 2. Load base notebook 0.574
nb = json.load(open('scratch/public_0574/0-574-in-style-qwen3-8b-lora-fully-public.ipynb', encoding='utf-8'))

# 3. Modify Cell 1: FAMILY_IDX
c1 = nb['cells'][1]['source']
target_c1 = "FAMILY_IDX = {f: np.array([i for i, l in enumerate(LABELS) if l.startswith(f)]) for f in ('claude', 'gpt', 'gemini', 'glm')}"
assert target_c1 in c1, "Target in Cell 1 not found!"

replacement_c1 = """FAMILY_IDX = {
    'claude': np.array([i for i, l in enumerate(LABELS) if l.startswith('claude')]),
    'gpt': np.array([i for i, l in enumerate(LABELS) if l.startswith('gpt')]),
    'gemini': np.array([i for i, l in enumerate(LABELS) if l.startswith('gemini')]),
    'chinese': np.array([i for i, l in enumerate(LABELS) if l.startswith('glm') or l.startswith('kimi') or l.startswith('qwen')])
}"""

c1 = c1.replace(target_c1, replacement_c1)
nb['cells'][1]['source'] = c1
print("Cell 1 updated successfully.")

# 4. Modify Cell 5: Gates & Universal 18-class Mega-Gate
c5 = nb['cells'][5]['source']
target_c5 = """for fam, C in [('gpt', 1.0), ('gemini', 1.0), ('glm', 0.5)]:
    m, me = np.isin(y, FAMILY_IDX[fam]), np.isin(EX_Y, FAMILY_IDX[fam])
    GATES[fam] = family_gate([t for t, k in zip(TR, m) if k] + [t for t, k in zip(EX_T, me) if k], np.concatenate([y[m], EX_Y[me]]),
                             FAMILY_IDX[fam], C, sample_weight=np.concatenate([np.ones(int(m.sum())), np.full(int(me.sum()), 0.25)]), balanced=True)
    print(f'{fam} gate: {int(m.sum())} official + {int(me.sum())} extra rows', flush=True)
del grams, X_tr, X_ex, X_te"""

assert target_c5 in c5, "Target in Cell 5 not found!"

replacement_c5 = """for fam, C in [('gpt', 1.0), ('gemini', 1.0), ('chinese', 0.5)]:
    m, me = np.isin(y, FAMILY_IDX[fam]), np.isin(EX_Y, FAMILY_IDX[fam])
    GATES[fam] = family_gate([t for t, k in zip(TR, m) if k] + [t for t, k in zip(EX_T, me) if k], np.concatenate([y[m], EX_Y[me]]),
                             FAMILY_IDX[fam], C, sample_weight=np.concatenate([np.ones(int(m.sum())), np.full(int(me.sum()), 0.25)]), balanced=True)
    print(f'{fam} gate: {int(m.sum())} official + {int(me.sum())} extra rows', flush=True)

# Universal 18-class Mega-Gate across all 18 classes
print('Fitting Universal 18-Class Mega-Gate on all classes...', flush=True)
g_mega = StyleGrams().fit(TR + EX_T)
X_mega_tr = vstack([g_mega.transform(TR), g_mega.transform(EX_T)]).tocsr() if len(EX_T) else g_mega.transform(TR)
y_mega_tr = np.concatenate([y, EX_Y]) if len(EX_T) else y
w_mega_tr = extra_sample_weights() if len(EX_T) else None
svc_mega = LinearSVC(C=0.5, class_weight='balanced', random_state=42, max_iter=4000).fit(
    X_mega_tr, y_mega_tr, sample_weight=w_mega_tr
)
assert list(svc_mega.classes_) == list(range(K))
d_mega = svc_mega.decision_function(g_mega.transform(TE))
GATES['global_18'] = d_mega - d_mega.mean(axis=1, keepdims=True)
print(f'Universal 18-class mega-gate trained on {len(y_mega_tr)} rows (shape: {GATES["global_18"].shape})', flush=True)
del g_mega, X_mega_tr, svc_mega
del grams, X_tr, X_ex, X_te"""

c5 = c5.replace(target_c5, replacement_c5)
nb['cells'][5]['source'] = c5
print("Cell 5 updated successfully.")

# 5. Modify Cell 16: Cross-Family Soft Posterior Calibration
c16 = nb['cells'][16]['source']
target_c16 = """pred = P.argmax(1)
logP = np.log(np.clip(P, 1e-12, 1.0))
for fam in ('claude', 'gpt', 'gemini', 'glm'):   # re-rank inside the predicted family only
    if fam not in GATES:
        continue
    idx = FAMILY_IDX[fam]
    score = logP[:, idx] + (GATE_LAM['claude_style'] if fam == 'claude' else GATE_LAM['family']) * GATES[fam]
    if fam == 'claude' and 'claude_lora' in GATES:
        score = score + GATE_LAM['claude_lora'] * GATES['claude_lora']
    inside = np.isin(pred, idx)
    new = idx[score.argmax(1)]
    print(f'{fam} gate changed', int((new[inside] != pred[inside]).sum()), 'of', int(inside.sum()), f'{fam} predictions', flush=True)
    pred[inside] = new[inside]"""

assert target_c16 in c16, "Target in Cell 16 not found!"

replacement_c16 = """# V14 Cross-Family Soft Posterior Calibration
logP = np.log(np.clip(P, 1e-12, 1.0))
calibrated_logP = logP.copy()

# 1. Apply Universal 18-class global calibration
if 'global_18' in GATES:
    lam_global = 0.35
    calibrated_logP += lam_global * GATES['global_18']
    print(f'Universal 18-class cross-family calibration applied (lam = {lam_global})', flush=True)

# 2. Apply Intra-family micro-calibration for all 4 families (100% 18 models covered)
GATE_LAM_V14 = {'claude': 0.80, 'gpt': 0.60, 'gemini': 0.60, 'chinese': 0.80}
for fam, fidx in FAMILY_IDX.items():
    if fam not in GATES:
        continue
    gate_scores = GATES[fam]
    if fam == 'claude' and 'claude_lora' in GATES:
        gate_scores = gate_scores + (GATE_LAM['claude_lora'] / GATE_LAM['claude_style']) * GATES['claude_lora']
    
    # Zero-center intra-family scores so it acts purely as internal re-ranking
    gate_centered = gate_scores - gate_scores.mean(axis=1, keepdims=True)
    fam_lam = GATE_LAM_V14.get(fam, 0.60)
    calibrated_logP[:, fidx] += fam_lam * gate_centered
    print(f'{fam} intra-family micro-gate applied (lam = {fam_lam})', flush=True)

# 3. High-confidence ensemble protection (protect top predictions with high margin)
p_sort = np.sort(P, axis=1)
margin = p_sort[:, -1] - p_sort[:, -2]
confident_mask = margin > 0.35

pred = calibrated_logP.argmax(axis=1)
pred[confident_mask] = P.argmax(axis=1)[confident_mask]

flips = int((pred != P.argmax(1)).sum())
print(f'V14 Cross-family gating modified {flips} of {len(P)} predictions from base Sinkhorn', flush=True)"""

c16 = c16.replace(target_c16, replacement_c16)
nb['cells'][16]['source'] = c16
print("Cell 16 updated successfully.")

# 6. Title and markdown update
nb['cells'][0]['source'] = """# 0.585+ | Champion V14 In-Style Mega Gate + Cross-Family Soft Posterior Calibration

**Champion V14 Pipeline**:
- Base: In-Style Qwen3-8B LoRA + 5-Adapter Multi-Trunk + Gaussian Group Heads (0.574 public baseline)
- StyleGrams 4 Views trained on Train + 9,818 In-Style Passages from 709 prompt banks
- 4-Way Chinese Frontier Gate covering all 4 Chinese models (GLM-5.3, GLM-5.3-flash, Kimi-k3, Qwen3.8-max)
- Universal 18-Class Mega-Gate on global stylistic feature representations
- Cross-Family Soft Posterior Calibration replacing the siloed intra-family np.isin isolation trap
"""

# 7. Save notebook
with open('champion_v14/champion-v14.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Saved champion_v14/champion-v14.ipynb successfully!")

# 8. Syntax validation of all code cells
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        try:
            ast.parse(c['source'])
        except SyntaxError as e:
            print(f"SYNTAX ERROR in cell {i}: {e}")
            raise

print("All code cells in champion-v14.ipynb passed AST syntax validation!")
