import json
import copy
import os

# Create champion_v52 directory
os.makedirs('champion_v52', exist_ok=True)

# Load champion_v51 notebook as clean baseline
with open('champion_v51/champion-v51.ipynb', 'r', encoding='utf-8') as f:
    v51_nb = json.load(f)

v52_nb = copy.deepcopy(v51_nb)

# 1. Update Cell 0: Title & Strategy
v52_nb['cells'][0]['source'] = [
    "# 0.700+ Global Gold Zone Frontier | Champion V52 Dual-Titan Sovereign\n",
    "\n",
    "**Champion V52 Dual-Titan Sovereign** represents our second overnight strike paired with V51, specifically engineered to maximize orthogonal diversity and push beyond our 0.659 high watermark towards the **0.700+ Global Gold Zone**:\n",
    "\n",
    "1. **ModernBERT-Large Deepening (w=2.4, Context 8192):**\n",
    "   - Boosts ModernBERT-Large from w=2.0 to w=2.4 to inject deeper 8192-token bidirectional contextual representations, harmonizing with Qwen 14B/8B causal attention.\n",
    "2. **Meta Llama-3.1-8B Calibrated Parity (w=2.0, dp: True, No Gauss):**\n",
    "   - Tunes Llama-3.1-8B to w=2.0 (in parity with Qwen 8B), maximizing clean Western tokenization diversity without prompt over-smoothing.\n",
    "3. **Adaptive LogSumExp Sibling Gate (Margin 0.36, Strength 0.30):**\n",
    "   - Expands the within-family sibling tie-breaking gate with margin threshold 0.36 and scaled strength 0.30 across Claude, Gemini, GLM, and GPT.\n",
    "   - Stamatatos-inspired stylometric fusion preserves exact family probability mass ($< 10^{-16}$ deviation) with zero cross-provider leakage.\n",
    "4. **ATLAS Likelihood Turbo (w=0.25, Margin 0.36):**\n",
    "   - Maximizes ATLAS 8B likelihood guidance to 25% on borderline passages (margin < 0.36).\n",
    "5. **Refined Sinkhorn Damping (0.42, Target Shrinkage 0.30):**\n",
    "   - Calibrates Sinkhorn damping to preserve sharp model confidence while aligning global marginals.\n",
    "\n",
    "Paired with Champion V51 for an exhaustive overnight parallel probe into the Global Gold Zone!\n"
]

# 2. Update Cell 6: infer.py with tuned Sibling Gate (margin 0.36, strength 0.30, diagnostics v52)
infer_code = ''.join(v51_nb['cells'][6]['source'])

infer_code = infer_code.replace("with open(_out.with_name('v51_diagnostics.json'), 'w') as _fo:", "with open(_out.with_name('v52_diagnostics.json'), 'w') as _fo:")
infer_code = infer_code.replace("'method': 'v51_family_sovereign_pure_apex_sibling_mass_conserved',", "'method': 'v52_dual_titan_sovereign_sibling_mass_conserved',")
infer_code = infer_code.replace("log('Champion V51 Pure Bayesian typography + Sibling Gate completed", "log('Champion V52 Pure Bayesian typography + Sibling Gate completed")

# Tune sibling gate strength and margin in infer_code:
old_sib_gate = """        _f_conf_gate = np.clip((0.35 - _f_margin) / 0.35, 0.0, 1.0)
        _f_strength = 0.25 * _f_conf_gate"""

new_sib_gate = """        _f_conf_gate = np.clip((0.36 - _f_margin) / 0.36, 0.0, 1.0)
        _f_strength = 0.30 * _f_conf_gate"""

assert old_sib_gate in infer_code, "Old sibling gate parameters not found in infer_code!"
infer_code = infer_code.replace(old_sib_gate, new_sib_gate)

v52_nb['cells'][6]['source'] = [line + '\n' for line in infer_code.strip().split('\n')]

# 3. Update Cell 11: Markdown Members description
v52_nb['cells'][11]['source'] = [
    "## 6. Members and run\n",
    "Champion V52 Dual-Titan Sovereign: Qwen3-14B (w=3.0, gauss=0.70) + Meta Llama-3.1-8B (w=2.0, dp=True, no gauss) + ModernBERT-Large (w=2.4) + Nishant Qwen3-14B (w=2.5) + Qwen3-8B (w=2.0, gauss=0.44, dp=True) + DeBERTa-v3-large (w=1.0) + Qwen2.5-14B (w=1.0) + ATLAS 25% (margin 0.36) + Damped Sinkhorn 0.42 (target mix 0.30) + Hierarchical Beta-Bernoulli Typography (37 markers) + LogSumExp Family Sibling Residual Gate (margin 0.36, strength 0.30).\n"
]

# 4. Update Cell 12: Code defining MODELS and execution with MB w=2.4, Llama w=2.0, ATLAS w=0.25 margin 0.36, Sinkhorn 0.42/0.30
cell12_code = ''.join(v51_nb['cells'][12]['source'])

# Update ModernBERT weight
cell12_code = cell12_code.replace('{"name": "mb_v5", "type": "enc", "ckpt": MODERNBERT, "w": 2.0},', '{"name": "mb_v5", "type": "enc", "ckpt": MODERNBERT, "w": 2.4},')

# Update Llama weight
cell12_code = cell12_code.replace('"w": 1.8, "bs": 16, "max_len": 384, "twin": 0.35, "dp": True})', '"w": 2.0, "bs": 16, "max_len": 384, "twin": 0.35, "dp": True})')
cell12_code = cell12_code.replace("GOLDEN RATIO w=1.8 + dp=True (No Gauss Over-smoothing)", "TITAN PARITY w=2.0 + dp=True (No Gauss Over-smoothing)")

# Update ATLAS and Sinkhorn parameters
cell12_code = cell12_code.replace("'--atlas_w', '0.24', '--atlas_margin', '0.35',", "'--atlas_w', '0.25', '--atlas_margin', '0.36',")
cell12_code = cell12_code.replace("'--sinkhorn', '0.45', '--sinkhorn_target_mix', '0.35',", "'--sinkhorn', '0.42', '--sinkhorn_target_mix', '0.30',")

v52_nb['cells'][12]['source'] = [line + '\n' for line in cell12_code.strip().split('\n')]

# Save notebook
output_nb_path = 'champion_v52/champion-v52.ipynb'
with open(output_nb_path, 'w', encoding='utf-8') as f:
    json.dump(v52_nb, f, indent=1)
print(f"Generated {output_nb_path} with {len(v52_nb['cells'])} cells.")

# Create kernel-metadata.json
with open('champion_v51/kernel-metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

meta['id'] = 'baonguyenduyquoc/name-that-llm-champion-v52-dual-titan-sovereign'
meta['title'] = 'Name That LLM Champion V52 Dual Titan Sovereign'
meta['code_file'] = 'champion-v52.ipynb'

output_meta_path = 'champion_v52/kernel-metadata.json'
with open(output_meta_path, 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)
print(f"Generated {output_meta_path} successfully.")

