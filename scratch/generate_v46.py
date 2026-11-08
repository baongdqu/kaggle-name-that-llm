import json

with open('champion_v44/champion-v44.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Title cell (cell 0)
nb['cells'][0]['source'] = [
    "# 0.665+ -> 0.675+ (Top 30 Grandmaster Frontier) | Champion V46 Apex Titan Hepta-Gate\n",
    "**Team: Mohamed Morsi (Bao Ng) | Target Top 30 Global Leaderboard | Current Record LB 0.658**\n",
    "\n",
    "### 🏆 Key Architectural Breakthroughs in Champion V46:\n",
    "1. **Dual-Titan Causal Backbone Amplification:**\n",
    "   - Meta Llama-3.1-8B (5-Fold LoRA, w=2.8, 128k BPE vocabulary, RoPE theta 500k).\n",
    "   - Alibaba Qwen-3-14B (5-Fold LoRA w=2.8, gauss=0.70) + Qwen-3-8B (5-Fold w=1.8, gauss=0.44).\n",
    "   - Nishant Qwen 14B (w=1.8) + Qwen2.5-14B (w=1.0).\n",
    "2. **Dual Bidirectional Encoders (ModernBERT w=3.0 SOTA / DeBERTa w=1.6):**\n",
    "   - Provides orthogonal bidirectional representations across the full context window.\n",
    "3. **Hepta-Family Micro-Gates (7 Sibling Precision Gates):**\n",
    "   - Gemini Thought System Gate (`gemini-3.8-flash` vs `gemini-3.7-flash`)\n",
    "   - GLM CJK Punctuation Gate (`glm-5.3` vs `glm-5.3-flash`)\n",
    "   - Claude Spaced Em-dash & Colon Header Gate (`claude-opus-5` vs `claude-opus-4-6`)\n",
    "   - OpenAI Straight Quotes Gate (`gpt-5.5` discriminator)\n",
    "   - OpenAI Astra Comma-Density Gate (`gpt-6-astra` vs `luna`/`sol`/`terra`)\n",
    "   - Kimi vs Qwen Chinese Frontier Gate (Em-dash & Semicolon signature: 0.86 vs 0.05)\n",
    "   - OpenAI Sol Curly Quote Gate (`\\u201c` / `\\u201d` discriminates `gpt-5.6-sol`)\n",
    "4. **Adaptive Class-Specific Anchor-Protected Kuhn-Munkres Hungarian Matching:**\n",
    "   - High-sensitivity classes protected at Delta >= 0.24, P >= 0.42.\n",
    "   - Dense sibling clusters protected at Delta >= 0.34, P >= 0.50.\n"
]

# Update cell 9 (pipeline.py)
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

gate_vars_to_find = "astra_boosts = 0"
new_gate_vars = "astra_boosts = 0\nkimi_boosts = 0\nqwen_boosts = 0\nsol_boosts = 0"
cell9_src = cell9_src.replace(gate_vars_to_find, new_gate_vars)

idx_g5 = cell9_src.find("# 5. OpenAI Astra statistical comma-density gate:")
idx_end_g5 = cell9_src.find("P /= P.sum(1, keepdims=True)\nlog(f'sibling micro-gates:")
cell9_src = cell9_src[:idx_g5] + new_hepta_gates + "\n" + cell9_src[idx_end_g5:]

old_log = "log(f'sibling micro-gates: applied {g38_boosts} Gemini, {glm_boosts} GLM, {claude_boosts} Claude, {gpt_boosts} GPT5.5, {astra_boosts} Astra')"
new_log = "log(f'sibling hepta-gates: applied {g38_boosts} Gemini, {glm_boosts} GLM, {claude_boosts} Claude, {gpt_boosts} GPT5.5, {astra_boosts} Astra, {kimi_boosts} Kimi, {qwen_boosts} Qwen, {sol_boosts} Sol')"
cell9_src = cell9_src.replace(old_log, new_log)

# Update Anchor Hungarian logic with Adaptive Class Margins
old_hungarian_anchor = "    is_anchor = (margins >= 0.30) & (top_p >= 0.48)"
new_hungarian_anchor = """    # Class-adaptive thresholds for anchor protection
    DELTA_THRESHOLDS = np.full(K, 0.30, dtype=np.float32)
    P_THRESHOLDS = np.full(K, 0.48, dtype=np.float32)
    for name in ['gpt-5.5', 'kimi-k3', 'gemini-3.1-pro-preview', 'claude-fable-5', 'glm-5.3-flash']:
        idx_s = LABELS.index(name)
        DELTA_THRESHOLDS[idx_s] = 0.24
        P_THRESHOLDS[idx_s] = 0.42
    for name in ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'claude-opus-4-6', 'claude-opus-5']:
        idx_s = LABELS.index(name)
        DELTA_THRESHOLDS[idx_s] = 0.34
        P_THRESHOLDS[idx_s] = 0.50
    is_anchor = (margins >= DELTA_THRESHOLDS[top_cls]) & (top_p >= P_THRESHOLDS[top_cls])"""

cell9_src = cell9_src.replace(old_hungarian_anchor, new_hungarian_anchor)
nb['cells'][9]['source'] = [line + '\n' for line in cell9_src.splitlines()]

# Update cell 11 (ModernBERT w=3.0)
cell11_src = ''.join(nb['cells'][11]['source'])
cell11_src = cell11_src.replace("dict(name='modernbert', type='enc', ckpt=MODERNBERT, w=2.6)", "dict(name='modernbert', type='enc', ckpt=MODERNBERT, w=3.0)")
nb['cells'][11]['source'] = [line + '\n' for line in cell11_src.splitlines()]

with open('champion_v46/champion-v46.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("SUCCESS: champion_v46/champion-v46.ipynb generated successfully!")

