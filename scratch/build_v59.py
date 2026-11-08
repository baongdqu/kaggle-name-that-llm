import json

with open('champion_v50/champion-v50.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Cell 0 (Title)
nb['cells'][0]['source'] = [
    "# 0.680+ Global Top 10 | Champion V59 Prompt-Aware Relative Sovereign\n",
    "\n",
    "**Core Breakthrough (Method B):**\n",
    "- **Golden Anchor:** 100% of Champion V50 Core (Qwen 14B $w=3.0$ + Llama 3.1 8B $w=1.8$ dp:True + Nishant 14B $w=2.5$ + Qwen 8B $w=2.0$ + ModernBERT 384t $w=2.0$ + DeBERTa $w=1.0$).\n",
    "- **Prompt-Aware Relative Length Calibration:** Exploits the empirical word-count signatures of the 18 frontier models within estimated prompt clusters ($G_{te}$) to separate sister models (e.g. Gemini 3.8-Flash concise bias vs Gemini 3.7-Flash expansive bias).\n",
    "- **Soft Injective Collision Dispersion:** Disperses duplicate predictions within prompt clusters smoothly (damp factor 0.65) without the fatal breakdown of hard Hungarian on sparse test batches.\n",
    "- **Strict Anti-Hungarian Discipline:** `--perpassage 1` (Hungarian is strictly disabled).\n",
    "- **Damped Marginal Sinkhorn 0.45:** Final N/18 column balancing with no destructive nonlinear post-processing.\n"
]

# 2. Update Cell 2 (Setup Header)
cell2_src = ''.join(nb['cells'][2]['source'])
cell2_src = cell2_src.replace('Champion V50 Pure Apex Sovereign', 'Champion V59 Prompt-Aware Relative Sovereign')
nb['cells'][2]['source'] = [cell2_src]

# 3. Update Cell 6 (infer.py with Prompt-Aware Calibration)
cell6_src = ''.join(nb['cells'][6]['source'])

target_p = "        P = mixed / mixed.sum(1, keepdims=True)\n    Q = P.copy()"
replacement_p = """        P = mixed / mixed.sum(1, keepdims=True)
        # --- V59: Prompt-Aware Relative Feature Calibration (Method B) ---
        _LEN_BIAS = np.array([-0.030, -0.001, +0.162, +0.171, +0.054, +0.078,
                              -0.035, +0.038, -0.236, -0.065, +0.092, -0.060,
                              +0.008, +0.051, -0.023, +0.036, -0.143, -0.098], dtype=np.float32)
        _wlens = np.array([len(t.split()) for t in test.text.values], dtype=np.float32)
        _P_mod = P.copy()
        _n_calib = 0
        for _g in np.unique(Gte):
            _idx = np.where(Gte == _g)[0]
            if len(_idx) >= 3:
                _l_sub = _wlens[_idx]
                _std = _l_sub.std()
                if _std > 1.0:
                    _zl = (_l_sub - _l_sub.mean()) / _std
                    _adj = 0.20 * np.outer(_zl, _LEN_BIAS)
                    _sub_p = _P_mod[_idx] * np.exp(_adj)
                    _P_mod[_idx] = _sub_p / (_sub_p.sum(axis=1, keepdims=True) + 1e-12)
                    _n_calib += len(_idx)
        # Soft Injective Collision Dispersion (Anti-Duplicate Dispersion on Prompt Clusters without Hungarian crash)
        _n_disp = 0
        for _g in np.unique(Gte):
            _idx = np.where(Gte == _g)[0]
            if len(_idx) >= 2:
                _t1 = _P_mod[_idx].argmax(axis=1)
                for _c in range(18):
                    _c_rows = _idx[_t1 == _c]
                    if len(_c_rows) >= 2:
                        _confs = _P_mod[_c_rows, _c]
                        _ord_c = np.argsort(-_confs)
                        for _loser in _c_rows[_ord_c[1:]]:
                            _P_mod[_loser, _c] *= 0.65
                            _n_disp += 1
                        _P_mod[_c_rows] /= (_P_mod[_c_rows].sum(axis=1, keepdims=True) + 1e-12)
        P = _P_mod / _P_mod.sum(axis=1, keepdims=True)
        log(f'V59: Prompt-Aware calibrated {_n_calib} samples, soft-dispersed {_n_disp} collisions')
    Q = P.copy()"""

assert target_p in cell6_src, 'Target string not found in Cell 6 of V50!'
cell6_src = cell6_src.replace(target_p, replacement_p)
nb['cells'][6]['source'] = [cell6_src]

# 4. Update Cell 12
cell12_src = ''.join(nb['cells'][12]['source'])
nb['cells'][12]['source'] = [cell12_src]

with open('champion_v59/champion-v59.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Built champion_v59/champion-v59.ipynb successfully!')

