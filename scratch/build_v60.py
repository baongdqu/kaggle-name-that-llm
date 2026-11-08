import json

with open('champion_v59/champion-v59.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Cell 0 (Title)
nb['cells'][0]['source'] = [
    "# 0.665+ Silver Medal Strike | Champion V60 Apex Synergy\n",
    "\n",
    "**Core Breakthrough (The Grand Synergy):**\n",
    "- **Golden Anchor:** 100% of Champion V59 Breakthrough Core (Score 0.662 Bronze Winner: Qwen 14B $w=3.0$ + Llama 3.1 8B $w=1.8$ dp:True + Nishant 14B $w=2.5$ + Qwen 8B $w=2.0$ + ModernBERT 384t $w=2.0$ + DeBERTa $w=1.0$).\n",
    "- **Prompt-Aware Relative Length Calibration:** Empirical word-count signatures $\\beta_m$ from `train.csv` inside estimated clusters $G_{te}$ to separate sister models (e.g. Gemini 3.8-Flash concise bias vs Gemini 3.7-Flash expansive bias).\n",
    "- **Soft Typography Guards:** Continuous 90% suppression of Gemini on em-dash `—` (0.09% train frequency) and Claude/GLM on curly apostrophe `’` (0.00% train frequency).\n",
    "- **Soft Injective Collision Dispersion:** Disperses duplicate predictions within prompt clusters smoothly (damp factor 0.60).\n",
    "- **Intra-Family Sibling LogSumExp Sharpening:** Conserves 100% of inter-family probability mass while sharpening sibling discrimination within Claude, Gemini, GPT, and GLM families (temperature 1.15).\n",
    "- **Strict Anti-Hungarian Discipline:** `--perpassage 1` (Hungarian is strictly disabled).\n",
    "- **Damped Marginal Sinkhorn 0.45:** Final N/18 column balancing with no destructive post-processing.\n"
]

# 2. Update Cell 2 (Setup Header)
cell2_src = ''.join(nb['cells'][2]['source'])
cell2_src = cell2_src.replace('Champion V59 Prompt-Aware Relative Sovereign', 'Champion V60 Apex Synergy')
nb['cells'][2]['source'] = [cell2_src]

# 3. Update Cell 6 (infer.py with Full V60 Apex Synergy)
cell6_src = ''.join(nb['cells'][6]['source'])

target_p = """        P = _P_mod / _P_mod.sum(axis=1, keepdims=True)
        log(f'V59: Prompt-Aware calibrated {_n_calib} samples, soft-dispersed {_n_disp} collisions')
    Q = P.copy()"""

replacement_p = """        # Soft Typography Guards (em-dash suppresses Gemini by 90%, curly apostrophe suppresses Claude/GLM by 90%)
        CLAUDE_IDX = [0, 1, 2, 3, 4]
        GEMINI_IDX = [5, 6, 7, 8]
        GLM_IDX = [9, 10]
        GPT_IDX = [11, 12, 13, 14, 15]
        for _i, _t in enumerate(test.text.values):
            if '—' in _t: _P_mod[_i, GEMINI_IDX] *= 0.10
            if '’' in _t: _P_mod[_i, CLAUDE_IDX + GLM_IDX] *= 0.10
        _P_mod /= (_P_mod.sum(axis=1, keepdims=True) + 1e-12)

        # Intra-Family Sibling LogSumExp Sharpening (Family mass mathematically conserved)
        _s_P = np.sort(_P_mod, 1)
        _ambig_mask = (_s_P[:, -1] - _s_P[:, -2]) < 0.35
        for _fam in [CLAUDE_IDX, GEMINI_IDX, GLM_IDX, GPT_IDX]:
            _fam_mass = _P_mod[:, _fam].sum(axis=1, keepdims=True)
            _fam_probs = _P_mod[:, _fam] / (_fam_mass + 1e-12)
            _fam_sharp = np.power(_fam_probs, 1.15)
            _fam_sharp /= (_fam_sharp.sum(axis=1, keepdims=True) + 1e-12)
            _P_mod[np.ix_(_ambig_mask, _fam)] = 0.75 * _P_mod[np.ix_(_ambig_mask, _fam)] + 0.25 * (_fam_mass[_ambig_mask] * _fam_sharp[_ambig_mask])

        P = _P_mod / (_P_mod.sum(axis=1, keepdims=True) + 1e-12)
        log(f'V60: Apex Synergy calibrated {_n_calib} samples, dispersed {_n_disp} collisions, sharpened {int(_ambig_mask.sum())} sibling ambiguities')
    Q = P.copy()"""

assert target_p in cell6_src, 'Target string not found in Cell 6 of V59!'
cell6_src = cell6_src.replace(target_p, replacement_p)
nb['cells'][6]['source'] = [cell6_src]

# 4. Write notebook
with open('champion_v60/champion-v60.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Built champion_v60/champion-v60.ipynb successfully!')

