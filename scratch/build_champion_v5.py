import json
import os

meta = {
  "id": "baonguyenduyquoc/name-that-llm-champion-v5-microstylometrics",
  "title": "Name That LLM - Champion V5 MicroStylometrics",
  "code_file": "champion-v5.ipynb",
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

with open('champion_v5/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)

nb = json.load(open('champion_v5/champion-v5.ipynb', encoding='utf-8'))
code = ''.join(nb['cells'][1]['source'])

# 1. Update style_lgb weight in W dictionary
old_w = "'style_lgb': 0.063"
new_w = "'style_lgb': 0.150"
assert old_w in code, "old_w not found!"
code = code.replace(old_w, new_w)

# 2. Enrich stylometric_features with 25+ micro-formatting features
old_stylo = """    f += [cnt.get(x, 0) / nw for x in FUNC]
    return f"""

new_stylo = """    f += [cnt.get(x, 0) / nw for x in FUNC]
    # --- Micro-formatting & typographic layout features ---
    paras = [p for p in t.split('\\n\\n') if p.strip()]
    p_lens = [len(p) for p in paras] or [0]
    f += [
        t.count('\\n\\n') / max(len(paras), 1),
        t.count('\\n') / nc,
        len(paras),
        float(np.mean(p_lens)) / nc,
        float(np.std(p_lens)) / nc,
        t.count('—') / nw,
        t.count('–') / nw,
        t.count('…') / nw,
        t.count('...') / nw,
        len(re.findall(r'\\*\\*.*?\\*\\*', t)) / nw,
        len(re.findall(r'(?m)^-\\s+', t)) / max(len(sents), 1),
        len(re.findall(r'(?m)^\\*\\s+', t)) / max(len(sents), 1),
        len(re.findall(r'(?m)^\\d+\\.\\s+', t)) / max(len(sents), 1),
        t.count('```') / max(len(sents), 1),
        len(re.findall(r'`[^`]+`', t)) / nw,
        len(re.findall(r'(?m)^>\\s+', t)) / max(len(sents), 1),
        (t.count('“') + t.count('”')) / nw,
        t.count('"') / nw,
        len(re.findall(r'(?i)\\bhowever\\b', t)) / max(len(sents), 1),
        len(re.findall(r'(?i)\\bfurthermore\\b', t)) / max(len(sents), 1),
        len(re.findall(r'(?i)\\bmoreover\\b', t)) / max(len(sents), 1),
        len(re.findall(r'(?i)\\btherefore\\b', t)) / max(len(sents), 1),
        len(re.findall(r'(?i)\\bspecifically\\b', t)) / max(len(sents), 1),
        len(re.findall(r'(?i)\\bindeed\\b', t)) / max(len(sents), 1)
    ]
    return f"""

assert old_stylo in code, "old_stylo not found!"
code = code.replace(old_stylo, new_stylo)

# 3. Add Intra-Cluster Gate Centering before inference gates
cluster_gate_code = """
# ------------------------------------------------ Intra-Cluster Gate Mean-Centering
# Remove prompt-level topic shifts across Gate decision functions
if len(TE) >= 2 and 'cl' in globals():
    try:
        for _gate_var in ['DC_TE', 'DC_TE_OPENAI', 'DC_TE_GEMINI', 'DC_TE_GLM']:
            if _gate_var in globals() and globals()[_gate_var] is not None:
                _arr = globals()[_gate_var]
                for _cid in np.unique(cl):
                    _cmask = (cl == _cid)
                    if _cmask.sum() >= 2:
                        _arr[_cmask] -= _arr[_cmask].mean(axis=0)
        print('Applied intra-cluster mean-centering across all 4 Gate decision functions!', flush=True)
    except Exception as e:
        print('Cluster gate centering skipped:', e, flush=True)
"""

pred_anchor = "pred = P.argmax(1)"
assert pred_anchor in code, "pred_anchor not found!"
code = code.replace(pred_anchor, cluster_gate_code + "\n" + pred_anchor)

nb['cells'][1]['source'] = code
with open('champion_v5/champion-v5.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("SUCCESS: champion-v5.ipynb created successfully!")
