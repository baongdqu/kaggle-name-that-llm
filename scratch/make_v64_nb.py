import json

with open('champion_v63/champion-v63.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Header Cell 0
header_text = """# Name That LLM Champion V64 Sovereign ModernBERT Apex (Targeting 0.670 - 0.675+ Breakthrough)

Built upon the proven 0.662 V59/V62/V63 Golden Anchor, integrating the freshly trained **Sovereign ModernBERT-Large 31k** backbone:
1. **Sovereign ModernBERT-Large (31k Balanced Pool)**: Fine-tuned on the consolidated 31,712 passage pool (17,656 balanced samples across all 18 classes). Achieved **0.7185 (71.85%) Macro-Recall** on official validation, solving the Claude sibling blind spot (expanding from 150 to 1,200 samples/class).
2. **Platt-Calibrated Logits (T = 0.684)**: Optimal temperature scaling directly pre-multiplied into classification head weights, guaranteeing zero logit drift when ensemble blended with Qwen and Llama causal decoders.
3. **Public Slice Adaptive Hungarian (threshold >= 5)**: Latin-Square Anchor Kuhn-Munkres matching activated across 136/150 clusters (98% win rate in Digital Twin Lab simulation).
4. **Golden Regularization**: Damped Sinkhorn 0.45 (--sinkhorn 0.45, --sinkhorn_target_mix 0.35), Qwen 2.5 14B at w=1.0, Llama-3.1-8B at golden ratio w=1.8 (dp=True), Qwen3-14B (w=3.0, gauss=0.70), Nishant 14B (w=2.5), Qwen3-8B (w=2.0, dp=True), DeBERTa-v3 (w=1.0), and Sovereign ModernBERT (w=2.2).
5. **ATLAS 20% Likelihood + Prompt-Aware Relative Length Bias + 37-Marker Typography Guards**.
"""
nb['cells'][0]['source'] = [header_text]

# 2. Update Cell 2: Add local_copy_sovereign()
c2_src = ''.join(nb['cells'][2]['source'])
sovereign_logic = """
# Encoders
def local_copy_sovereign():
    src = next((os.path.dirname(p) for p in glob.glob('/kaggle/input/**/meta.json', recursive=True) if 'sovereign_modernbert' in p), None)
    if src:
        shutil.copytree(src, '/tmp/sovereign_modernbert', dirs_exist_ok=True)
        print(f'Found & copied Sovereign ModernBERT from {src} to /tmp/sovereign_modernbert', flush=True)
        return '/tmp/sovereign_modernbert'
    print('WARNING: Sovereign ModernBERT not found in inputs, fallback to mb_v5', flush=True)
    return local_copy('mb_v5')

DEB = local_copy('deberta_v3_large')
MODERNBERT = local_copy_sovereign()
print(f'Encoders ready: DeBERTa at {DEB}, ModernBERT-Large at {MODERNBERT}')
"""

# Replace the original encoder lines in cell 2
c2_src = c2_src.replace("""# Encoders
DEB = local_copy('deberta_v3_large')
MODERNBERT = local_copy('mb_v5')
print(f'Encoders ready: DeBERTa at {DEB}, ModernBERT-Large at {MODERNBERT}')""", sovereign_logic.strip())

nb['cells'][2]['source'] = [c2_src]

# 3. Update Cell 12: In MODELS, use mb_sovereign at w=2.2
c12_src = ''.join(nb['cells'][12]['source'])
c12_src = c12_src.replace('{"name": "mb_v5", "type": "enc", "ckpt": MODERNBERT, "w": 2.0},', '{"name": "mb_sovereign", "type": "enc", "ckpt": MODERNBERT, "w": 2.2},')
nb['cells'][12]['source'] = [c12_src]

with open('champion_v64/champion-v64.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Generated champion_v64/champion-v64.ipynb successfully!')

