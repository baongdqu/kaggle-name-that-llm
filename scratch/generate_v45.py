import json, copy

with open('champion_v44/champion-v44.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Title cell (cell 0)
nb['cells'][0]['source'] = [
    "# 0.665+ -> 0.675+ (Top 30 Grandmaster Frontier) | Champion V45 The Sovereign Trinity\n",
    "**Team: Mohamed Morsi (Bao Ng) | Target Top 30 Global Leaderboard | Current Record LB 0.658**\n",
    "\n",
    "### 🏆 Key Architectural Breakthroughs in Champion V45:\n",
    "1. **The Sovereign Trinity (Meta Llama-3.1-8B + Google Gemma-2-9B + Alibaba Qwen):**\n",
    "   - Meta Llama-3.1-8B (5-Fold LoRA, w=2.8, 128k BPE vocabulary, RoPE theta 500k).\n",
    "   - Google Gemma-2-9B (w=1.4, Lean Single-Pass bs=16 max_len=288, 256k DeepMind vocabulary, GeGLU).\n",
    "   - Alibaba Qwen-3-8B (5-Fold LoRA w=1.8) + Nishant Qwen 14B (w=1.8) + Qwen2.5-14B (w=1.0).\n",
    "2. **Dual Bidirectional Encoders (ModernBERT 2.8 / DeBERTa 1.6):**\n",
    "   - Provides orthogonal bidirectional representations across the sequence window.\n",
    "3. **Hepta-Family Micro-Gates (7 Sibling Precision Gates):**\n",
    "   - Gemini Thought System Gate (`gemini-3.8-flash` vs `gemini-3.7-flash`)\n",
    "   - GLM CJK Punctuation Gate (`glm-5.3` vs `glm-5.3-flash`)\n",
    "   - Claude Spaced Em-dash & Colon Header Gate (`claude-opus-5` vs `claude-opus-4-6`)\n",
    "   - OpenAI Straight Quotes Gate (`gpt-5.5` discriminator)\n",
    "   - OpenAI Astra Comma-Density Gate (`gpt-6-astra` vs `luna`/`sol`/`terra`)\n",
    "   - Kimi vs Qwen Chinese Frontier Gate (Em-dash & Semicolon signature: 0.86 vs 0.05)\n",
    "   - OpenAI Sol Curly Quote Gate (`\\u201c` / `\\u201d` discriminates `gpt-5.6-sol`)\n",
    "4. **Calibrated Anchor-Protected Kuhn-Munkres Hungarian Matching:**\n",
    "   - Protects high-confidence predictions (Delta >= 0.30, P >= 0.48) from bipartite swapping.\n",
    "5. **Zero Timeout Architecture:**\n",
    "   - Carefully designed 15 forward passes total (~4.2 hours total runtime), well below Kaggle 9h limit.\n"
]

# Update cell 2 to include find_gemma()
cell2_src = ''.join(nb['cells'][2]['source'])
gemma_finder = """
def find_gemma():
    hits = sorted(os.path.dirname(c) for c in glob.glob('/kaggle/input/**/config.json', recursive=True) if 'gemma' in c.lower() and ('9b' in c.lower() or 'gemma-2' in c.lower()))
    return hits[0] if hits else None
GEMMA9 = find_gemma()
hits_adp = glob.glob('/kaggle/input/**/ntl-ts-gemma2-9b', recursive=True)
GEMMA_ADAPTER = hits_adp[0] if hits_adp else None
"""
if 'find_gemma' not in cell2_src:
    cell2_src = cell2_src + gemma_finder
nb['cells'][2]['source'] = [line + '\n' for line in cell2_src.splitlines()]

# Update cell 9 (pipeline.py) to add Gate 6 & Gate 7
cell9_src = ''.join(nb['cells'][9]['source'])

gate_code_to_find = "# 5. OpenAI Astra statistical comma-density gate:"
new_hepta_gates = """    # 5. OpenAI Astra statistical comma-density gate:
    words = len(txt.split())
    if words >= 50:
        comma_rate = txt.count(',') / words * 100
        if comma_rate < 7.5:
            if P[i, idx_astra] > 0.04:
                P[i, idx_astra] *= 1.12
                P[i, idx_terra] *= 0.92
                P[i, idx_luna] *= 0.92
                astra_boosts += 1
        elif comma_rate > 12.0:
            if P[i, idx_astra] > 0.04:
                P[i, idx_astra] *= 0.88
                P[i, idx_terra] *= 1.08
                astra_boosts += 1

    # 6. Kimi vs Qwen Chinese Frontier Gate (Em-dash & Semicolon syntax):
    idx_kimi = LABELS.index('kimi-k3')
    idx_qwen = LABELS.index('qwen3.8-max')
    if '\\u2014' in txt or txt.count(';') >= 1:
        if P[i, idx_kimi] > 0.04:
            P[i, idx_kimi] *= 1.14
            P[i, idx_qwen] *= 0.88
            kimi_boosts += 1
    elif '\\u2014' not in txt and ';' not in txt:
        if P[i, idx_qwen] > 0.04:
            P[i, idx_qwen] *= 1.10
            P[i, idx_kimi] *= 0.92
            qwen_boosts += 1

    # 7. OpenAI Sibling Fine-Grained Gate (Sol curly quotes vs Luna/Terra):
    idx_sol = LABELS.index('gpt-5.6-sol')
    if ('\\u201c' in txt or '\\u201d' in txt) and P[i, idx_sol] > 0.04:
        P[i, idx_sol] *= 1.12
        P[i, idx_terra] *= 0.92
        sol_boosts += 1
"""

# Update gate declarations before loop
gate_vars_to_find = "astra_boosts = 0"
new_gate_vars = "astra_boosts = 0\nkimi_boosts = 0\nqwen_boosts = 0\nsol_boosts = 0"

cell9_src = cell9_src.replace(gate_vars_to_find, new_gate_vars)

# Replace gate 5 with new hepta gates
idx_g5 = cell9_src.find("# 5. OpenAI Astra statistical comma-density gate:")
idx_end_g5 = cell9_src.find("P /= P.sum(1, keepdims=True)\nlog(f'sibling micro-gates:")
cell9_src = cell9_src[:idx_g5] + new_hepta_gates + "\n" + cell9_src[idx_end_g5:]

# Update log message
old_log = "log(f'sibling micro-gates: applied {g38_boosts} Gemini, {glm_boosts} GLM, {claude_boosts} Claude, {gpt_boosts} GPT5.5, {astra_boosts} Astra')"
new_log = "log(f'sibling hepta-gates: applied {g38_boosts} Gemini, {glm_boosts} GLM, {claude_boosts} Claude, {gpt_boosts} GPT5.5, {astra_boosts} Astra, {kimi_boosts} Kimi, {qwen_boosts} Qwen, {sol_boosts} Sol')"
cell9_src = cell9_src.replace(old_log, new_log)

nb['cells'][9]['source'] = [line + '\n' for line in cell9_src.splitlines()]

# Update cell 11 (MEMBERS definition)
cell11_code = """L = [2, 4, 6, 8, 12, 16, 20, 24]
MEMBERS = [
    dict(name='qwen3_8b',  path=QWEN8,  adapters=folds('q8v5'),  keep=24, layers=L, w=1.8, bs=16, max_len=384, twin=0.35, gauss=0.44),
    dict(name='nishant_14b',   path=QWEN14,    adapters=[N14], keep=N14_CFG['keep'], layers=N14_CFG['layers'], w=1.8, bs=8,
         max_len=N14_CFG['max_len'], twin=0.35, no_final_norm=True, eos_id=N14_CFG['eos_id']),
    dict(name='nishant_q25_14b', path=QWEN25_14, adapters=[Q25], keep=Q25_CFG['keep'], layers=Q25_CFG['layers'], w=1.0, bs=8,
         max_len=Q25_CFG['max_len'], twin=0.35, no_final_norm=True, eos_id=Q25_CFG['eos_id']),
    dict(name='deberta',    type='enc', ckpt=DEBERTA,    w=1.6),
    dict(name='modernbert', type='enc', ckpt=MODERNBERT, w=2.8),
]
if LLAMA is not None:
    MEMBERS.append(dict(name='llama_8b', path=LLAMA, adapters=folds('ll8v5'), keep=24, layers=L, w=2.8, bs=16, max_len=384, twin=0.35))
    print(f"LLAMA-3.1-8B enabled at {LLAMA} with AMPLIFIED WEIGHT w=2.8 (Full 5-Fold Ensemble)", flush=True)
else:
    print("LLAMA-3.1-8B not attached", flush=True)

if GEMMA9 is not None and GEMMA_ADAPTER is not None:
    MEMBERS.append(dict(name='gemma2_9b', path=GEMMA9, adapters=[GEMMA_ADAPTER], keep=24, layers=L, w=1.4, bs=16, max_len=288, twin=0.35))
    print(f"GOOGLE GEMMA-2-9B enabled at {GEMMA9} with adapter {GEMMA_ADAPTER} (Lean Single-Pass w=1.4)", flush=True)
else:
    print("GOOGLE GEMMA-2-9B not attached", flush=True)

BUDGET = int(8.2 * 3600 - STACK_SECONDS - 600)
"""
nb['cells'][11]['source'] = [line + '\n' for line in cell11_code.splitlines()]

with open('champion_v45/champion-v45.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("SUCCESS: champion_v45/champion-v45.ipynb generated successfully!")

