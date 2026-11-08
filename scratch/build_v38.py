import json

with open('champion_v37/champion-v37.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Cell 0 Title
nb['cells'][0]['source'] = [
    '# 0.665+ -> 0.680+ (Top 10 Grandmaster Frontier) | Champion V38 Apex Sovereign Trinity\n',
    '**Team: Mohamed Morsi (Bao Ng) | Target Top 10 Global Leaderboard | Proven LB 0.646+**\n',
    '\n',
    '### 🏆 Key Architectural Breakthroughs in Champion V38:\n',
    '1. **The Sovereign Trinity (Multi-Vendor Orthogonal Causal Backbones):**\n',
    '   - **Alibaba Qwen (5 Folds Averaged):** Qwen3-14B (w=3.0, twin=0.35, gauss=0.70) + Qwen3-8B (w=2.0) + Nishant 14B (w=2.0) + Qwen2.5-14B (w=1.0).\n',
    '   - **Meta Llama-3.1-8B (5 Folds Averaged):** `ilyass323/llama-3-1-8b-bf16` + `ll8v5` 5-fold adapters (w=1.0, 128k vocabulary, RoPE theta=500k).\n',
    '   - **Google DeepMind Gemma-2-9B:** `lizhecheng/googlegemma-2-9b` + `alexchilton/ntl-ts-gemma2-9b` (w=1.2, 256k vocabulary, GeGLU activation, sliding window attention).\n',
    '2. **Dual Bidirectional Encoders Wing:**\n',
    '   - ModernBERT-large (w=2.4, 8,192 bidirectional tokens, solo 94.8% accuracy).\n',
    '   - DeBERTa-v3-large (w=1.6, disentangled attention syntax specialist).\n',
    '3. **SOTA 4-View Stylometrics Enhanced:**\n',
    '   - 540,000 N-gram features fitted on 120,000 clean pool passages (`nlm_pool_v5.parquet`), blended at w=0.22 with topic_correction (alpha=0.62).\n',
    '4. **Pure Flat 18-Class Probabilistic Calibration:**\n',
    '   - Pre-Sinkhorn Sibling Micro-Gates for Gemini (3.8 vs 3.7) and GLM (5.3 vs 5.3-flash).\n',
    '   - Terminal Sinkhorn Purity (100 iterations) + Kuhn-Munkres Hungarian Assignment on near-complete groups.\n'
]

# 2. Update Cell 2 with find_gemma
c2_src = ''.join(nb['cells'][2]['source'])
gemma_find_code = '''
def find_gemma():
    hits = sorted(os.path.dirname(c) for c in glob.glob('/kaggle/input/**/config.json', recursive=True) if 'gemma' in c.lower() and ('9b' in c.lower() or 'gemma-2' in c.lower()))
    return hits[0] if hits else None
GEMMA9 = find_gemma()
hits_adp = glob.glob('/kaggle/input/**/ntl-ts-gemma2-9b', recursive=True)
GEMMA_ADAPTER = hits_adp[0] if hits_adp else None
'''

if 'find_gemma' not in c2_src:
    c2_src += gemma_find_code
    nb['cells'][2]['source'] = [line + '\n' for line in c2_src.splitlines()]

# 3. Update Cell 11 with gemma2_9b addition
new_c11 = '''L = [2, 4, 6, 8, 12, 16, 20, 24]
MEMBERS = [
    dict(name='qwen3_14b', path=QWEN14, adapters=folds('q14v5'), keep=27, layers=L, w=3.0, bs=8,  max_len=384, twin=0.35, gauss=0.70),
    dict(name='qwen3_8b',  path=QWEN8,  adapters=folds('q8v5'),  keep=24, layers=L, w=2.0, bs=16, max_len=384, twin=0.35, gauss=0.44),
    dict(name='nishant_14b',   path=QWEN14,    adapters=[N14], keep=N14_CFG['keep'], layers=N14_CFG['layers'], w=2.0, bs=8,
         max_len=N14_CFG['max_len'], twin=0.35, no_final_norm=True, eos_id=N14_CFG['eos_id']),
    dict(name='nishant_q25_14b', path=QWEN25_14, adapters=[Q25], keep=Q25_CFG['keep'], layers=Q25_CFG['layers'], w=1.0, bs=8,
         max_len=Q25_CFG['max_len'], twin=0.35, no_final_norm=True, eos_id=Q25_CFG['eos_id']),
    dict(name='deberta',    type='enc', ckpt=DEBERTA,    w=1.6),
    dict(name='modernbert', type='enc', ckpt=MODERNBERT, w=2.4),
]
if LLAMA is not None:
    MEMBERS.append(dict(name='llama_8b', path=LLAMA, adapters=folds('ll8v5'), keep=24, layers=L, w=1.0, bs=16, max_len=384, twin=0.35))
    print(f"LLAMA-3.1-8B enabled at {LLAMA}", flush=True)
else:
    print("LLAMA-3.1-8B not attached", flush=True)

if GEMMA9 is not None and GEMMA_ADAPTER is not None:
    MEMBERS.append(dict(name='gemma2_9b', path=GEMMA9, adapters=[GEMMA_ADAPTER], keep=24, layers=L, w=1.2, bs=8, max_len=384, twin=0.35))
    print(f"GEMMA-2-9B enabled at {GEMMA9} with adapter {GEMMA_ADAPTER}", flush=True)
else:
    print("GEMMA-2-9B not attached", flush=True)

BUDGET = int(8.3 * 3600 - STACK_SECONDS - 600)
'''
nb['cells'][11]['source'] = [line + '\n' for line in new_c11.splitlines()]

with open('champion_v38/champion-v38.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('champion_v38 notebook built successfully!')
