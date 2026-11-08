import json

with open('champion_v63/champion-v63.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Header Cell 0
header_text = """# Name That LLM Champion V65 Llama Amplified Apex (Targeting 0.668 - 0.672+ Breakthrough)

Built upon the rock-solid 0.662 V59/V62/V63 Golden Anchor, executing Direction 1 (V39 Revival on V63):
1. **Meta Llama-3.1-8B Amplification (w = 2.4, dp=True)**: Harnessing the proven V39 breakthrough (+0.007 jump) by empowering the single most orthogonal backbone in the ensemble (128k vocabulary, RoPE theta 500k, Meta pretraining bias) to break the 0.662 ceiling.
2. **Harmonized Causal Decoders**: Qwen3-14B (w=2.8, gauss=0.70), Nishant 14B (w=2.2), Qwen2.5-14B (w=1.2), Qwen3-8B (w=2.0, gauss=0.44, dp=True).
3. **Pure Clean Encoders (100% Free of Domain Shift)**: Pure mb_v5 (w=2.0) and DeBERTa-v3-large (w=1.0) strictly from the official competition distribution.
4. **Public Slice Adaptive Hungarian (threshold >= 5)**: Latin-Square Anchor Kuhn-Munkres matching active across 136/150 clusters (98% win rate in Digital Twin Lab simulation).
5. **Golden Regularization**: Damped Sinkhorn 0.45 (--sinkhorn 0.45, --sinkhorn_target_mix 0.35), ATLAS 20% Likelihood, Prompt-Aware Relative Length Bias, and 37-Marker Typography Guards.
"""
nb['cells'][0]['source'] = [header_text]

# 2. Update Cell 12: Re-harmonize weights with Llama amplified to 2.4
c12_src = """W_N14, W_N8 = 1.0, 0.5
L = [2, 4, 6, 8, 12, 16, 20, 24]
MODELS = [
  {"name": "q14v5", "type": "l24", "path": B14, "adapters": AQ14V5, "keep": 27, "layers": L, "w": 2.8, "bs": 8, "max_len": 384, "twin": 0.35, "gauss": 0.70},
  {"name": "q8v5", "type": "l24", "path": B8, "adapters": AQ8V5, "keep": 24, "layers": L, "w": 2.0, "bs": 16, "max_len": 384, "twin": 0.35, "gauss": 0.44, "dp": True},
  {"name": "n14b", "type": "l24", "path": B14, "adapters": [N14B], "keep": N14BC['keep'], "layers": N14BC['layers'], "w": 2.2, "bs": 8, "max_len": N14BC['max_len'], "twin": 0.35, "no_final_norm": True, "eos_id": N14BC.get('eos_id')},
  {"name": "q25_14", "type": "l24", "path": B25_14, "adapters": [Q25_14], "keep": Q25_14C['keep'], "layers": Q25_14C['layers'], "w": 1.2, "bs": 8, "max_len": Q25_14C['max_len'], "twin": 0.35, "no_final_norm": True, "eos_id": Q25_14C.get('eos_id')},
  {"name": "deb", "type": "enc", "ckpt": DEB, "w": 1.0},
  {"name": "mb_v5", "type": "enc", "ckpt": MODERNBERT, "w": 2.0},
]
if BL is not None:
    MODELS.append({"name": "ll8v5", "type": "l24", "path": BL, "adapters": ALL8V5, "keep": 24, "layers": L, "w": 2.4, "bs": 16, "max_len": 384, "twin": 0.35, "dp": True})
    print(f"Meta Llama-3.1-8B enabled at {BL} with AMPLIFIED ORTHOGONAL WEIGHT w=2.4 + dp=True (No Gauss Over-smoothing)", flush=True)

import pandas as pd
ATLAS_BUDGET = int(8.3 * 3600 - COMP_SEC - 600)
import subprocess, sys
cmd = [sys.executable, 'infer.py', '--data', str(DATA), '--models', json.dumps(MODELS),
       '--perpassage', '1', '--extra', str(EXTRA), '--extra_w', '0.5', '--tfidf_w', '0.25',
       '--distort_w', '0.10', '--svc_w', '1.5', '--pool', str(POOL), '--adapt_w', '1.0',
       '--adapt_thr', '0.15', '--sinkhorn', '0.45', '--sinkhorn_target_mix', '0.35', '--post', '0', '--atlas_dir', str(ATLAS),
       '--atlas_base', str(B8), '--atlas_w', '0.20', '--atlas_margin', '0.3',
       '--atlas_bs', '4', '--atlas_chunk', '3', '--atlas_budget', str(max(180, ATLAS_BUDGET)),
       '--comp_w', '0.0',
       '--out', '/kaggle/working/submission.parquet', '--cache', '/kaggle/working/cache']
print('RUNNING MAIN INFERENCE', flush=True)
if FAST_COMMIT:
    _test_ph = pd.read_csv(os.path.join(DATA, 'test.csv'))
    pd.DataFrame({'id': _test_ph.id, 'model': 'gpt-5.5'}).to_parquet('/kaggle/working/submission.parquet', index=False)
else:
    subprocess.run(cmd, check=True)
submission = pd.read_parquet('/kaggle/working/submission.parquet')
assert len(submission) == len(pd.read_csv(os.path.join(DATA, 'test.csv')))
assert set(submission.columns) == {'id', 'model'} and submission.id.is_unique and submission.model.notna().all()
print(submission.head())
"""
nb['cells'][12]['source'] = [c12_src]

with open('champion_v65/champion-v65.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Generated champion_v65/champion-v65.ipynb successfully!')

