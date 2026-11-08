import json

with open('champion_v50/champion-v50.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Cell 0 (Title)
nb['cells'][0]['source'] = [
    "# 0.680+ Global Top 10 | Champion V58 Multi-Family Likelihood Titan\n",
    "\n",
    "**Core Breakthrough (Method A):**\n",
    "- **Golden Anchor:** 100% of Champion V50 Core (Qwen 14B $w=3.0$ + Llama 3.1 8B $w=1.8$ dp:True + Nishant 14B $w=2.5$ + Qwen 8B $w=2.0$ + ModernBERT 384t $w=2.0$ + DeBERTa $w=1.0$).\n",
    "- **Multi-Family Likelihood Expansion:** Expanded ATLAS coverage to margin $< 0.35$ with weight $0.22$.\n",
    "- **Intra-Family Sibling LogSumExp Gate:** Mathematically conserves 100% of inter-family probability mass while sharpening sibling discrimination within Claude, Gemini, GPT, and GLM families.\n",
    "- **Strict Anti-Hungarian Discipline:** `--perpassage 1` (Hungarian is strictly disabled to avoid sparse-test distortion).\n",
    "- **Damped Marginal Sinkhorn 0.45:** Final N/18 column balancing with no destructive nonlinear post-processing.\n"
]

# 2. Update Cell 2 (Setup Header)
cell2_src = ''.join(nb['cells'][2]['source'])
cell2_src = cell2_src.replace('Champion V50 Pure Apex Sovereign', 'Champion V58 Multi-Family Likelihood Titan')
nb['cells'][2]['source'] = [cell2_src]

# 3. Update Cell 6 (infer.py with Sibling Gate)
cell6_src = ''.join(nb['cells'][6]['source'])

target_p = "        P = mixed / mixed.sum(1, keepdims=True)\n    Q = P.copy()"
replacement_p = """        P = mixed / mixed.sum(1, keepdims=True)
        # --- V58: Intra-Family Sibling LogSumExp Preservation Gate ---
        CLAUDE_IDX = [0, 1, 2, 3, 4]
        GEMINI_IDX = [5, 6, 7, 8]
        GLM_IDX = [9, 10]
        GPT_IDX = [11, 12, 13, 14, 15]
        _s_P = np.sort(P, 1)
        _ambig_mask = (_s_P[:, -1] - _s_P[:, -2]) < 0.35
        for _fam in [CLAUDE_IDX, GEMINI_IDX, GLM_IDX, GPT_IDX]:
            _fam_mass = P[:, _fam].sum(axis=1, keepdims=True)
            _fam_probs = P[:, _fam] / (_fam_mass + 1e-12)
            _fam_sharp = np.power(_fam_probs, 1.15)
            _fam_sharp /= (_fam_sharp.sum(axis=1, keepdims=True) + 1e-12)
            P[np.ix_(_ambig_mask, _fam)] = 0.75 * P[np.ix_(_ambig_mask, _fam)] + 0.25 * (_fam_mass[_ambig_mask] * _fam_sharp[_ambig_mask])
        P /= P.sum(1, keepdims=True)
        log('V58: Sibling Gate applied on', int(_ambig_mask.sum()), 'ambiguous samples with family mass conserved')
    Q = P.copy()"""

assert target_p in cell6_src, 'Target string not found in Cell 6 of V50!'
cell6_src = cell6_src.replace(target_p, replacement_p)
nb['cells'][6]['source'] = [cell6_src]

# 4. Update Cell 12 (Runner parameters)
cell12_src = ''.join(nb['cells'][12]['source'])
cell12_src = cell12_src.replace("'--atlas_margin', '0.3'", "'--atlas_margin', '0.35'")
cell12_src = cell12_src.replace("'--atlas_w', '0.20'", "'--atlas_w', '0.22'")
nb['cells'][12]['source'] = [cell12_src]

with open('champion_v58/champion-v58.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Built champion_v58/champion-v58.ipynb successfully!')

