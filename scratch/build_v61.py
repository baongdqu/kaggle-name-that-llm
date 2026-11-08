import json
import os
import shutil

def build_v61():
    src_nb_path = 'scratch/0-664-hyperparameter-tuned-who-are-u/0-664-hyperparameter-tuned-who-are-u.ipynb'
    out_dir = 'champion_v61'
    os.makedirs(out_dir, exist_ok=True)
    out_nb_path = os.path.join(out_dir, 'champion-v61.ipynb')

    with open(src_nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Update Cell 0: Title & Strategy
    nb['cells'][0]['source'] = [
        "# Name That LLM Champion V61 GLM-4 SOTA Apex\n",
        "# Global Silver/Gold Breakthrough (Score 0.665+ Target)\n",
        "#\n",
        "# Architecture Highlights:\n",
        "# 1. GLM-4 9B Chinese Frontier Backbone (lucasboesen/name-that-llm-glm4-9b-trunc25) via Comp-Stack 15%\n",
        "# 2. Orthogonal ELECTRA-Large + DeskLib + DeBERTa-v3-large feature representations\n",
        "# 3. Judith007 SOTA Tuning: Sinkhorn 0.60 + Evidence Clip 1.2 (Rank 87)\n",
        "# 4. V59 Prompt-Aware Relative Length Calibration (Delta z_len * beta_m) inside clusters G_te\n",
        "# 5. Soft Injective Collision Dispersion (damp 0.65) + Strict Anti-Hungarian (--perpassage 1)\n",
        "# 6. Boosted Qwen 2.5 14B (w=2.0) and Golden Llama 3.1 8B (w=1.8 dp:True)\n",
        "\n",
        "print('Champion V61 GLM-4 SOTA Apex initialized.')\n"
    ]

    # 2. Update Cell 6 (infer.py): Inject V59 Prompt-Aware Relative Length Bias and Soft Collision Dispersion
    cell6_src = ''.join(nb['cells'][6]['source'])
    
    target_atlas_end = "        P = mixed / mixed.sum(1, keepdims=True)\n    Q = P.copy()"
    v61_calibration_injection = """        P = mixed / mixed.sum(1, keepdims=True)
    # --- V61: Prompt-Aware Relative Length Calibration & Soft Injective Dispersion ---
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
    P = _P_mod / (_P_mod.sum(axis=1, keepdims=True) + 1e-12)
    log(f'V61: Prompt-Aware calibrated {_n_calib} samples, soft-dispersed {_n_disp} collisions')
    Q = P.copy()"""

    assert target_atlas_end in cell6_src, 'Target ATLAS end string not found in Cell 6!'
    cell6_src = cell6_src.replace(target_atlas_end, v61_calibration_injection)
    nb['cells'][6]['source'] = [cell6_src]

    # 3. Update Cell 8 (comp_stack.py): Fix find_checkpoint for deberta directory name
    cell8_src = ''.join(nb['cells'][8]['source'])
    
    old_find_ckpt = """def find_checkpoint(model_type, hidden=None, dir_name=None):
    \"\"\"A checkpoint directory under /kaggle/input by model_type (+ hidden size) or by directory name.\"\"\"
    pattern = f'/kaggle/input/**/{dir_name}/config.json' if dir_name else '/kaggle/input/**/config.json'
    for cfg in sorted(glob.glob(pattern, recursive=True)):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        if dir_name or (c.get('model_type') == model_type and (hidden is None or c.get('hidden_size') == hidden)):
            return os.path.dirname(cfg)
    return None"""

    new_find_ckpt = """def find_checkpoint(model_type, hidden=None, dir_name=None):
    \"\"\"A checkpoint directory under /kaggle/input by model_type (+ hidden size) or by directory name.\"\"\"
    pattern = '/kaggle/input/**/config.json'
    for cfg in sorted(glob.glob(pattern, recursive=True)):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        if dir_name:
            d_clean = dir_name.lower().replace('-', '_')
            cfg_clean = os.path.dirname(cfg).lower().replace('-', '_')
            if d_clean in cfg_clean or ('deberta' in d_clean and 'deberta' in cfg_clean):
                return os.path.dirname(cfg)
        elif c.get('model_type') == model_type and (hidden is None or c.get('hidden_size') == hidden):
            return os.path.dirname(cfg)
    return None"""

    assert old_find_ckpt in cell8_src, 'Old find_checkpoint not found in Cell 8!'
    cell8_src = cell8_src.replace(old_find_ckpt, new_find_ckpt)
    nb['cells'][8]['source'] = [cell8_src]

    # 4. Update Cell 12: Boost Qwen 2.5 14B (w=2.0) and Llama 8B (w=1.8 dp:True)
    cell12_src = ''.join(nb['cells'][12]['source'])
    
    old_models_def = """  {"name": "ll8v5", "type": "l24", "path": BL, "adapters": ALL8V5, "keep": 24, "layers": L, "w": 1.0, "bs": 16, "max_len": 384, "twin": 0.35, "dp": True},
  {"name": "n14b", "type": "l24", "path": B14, "adapters": [N14B], "keep": N14BC['keep'], "layers": N14BC['layers'], "w": 2.5, "bs": 8, "max_len": N14BC['max_len'], "twin": 0.35, "no_final_norm": True, "eos_id": N14BC['eos_id']},
  {"name": "q25_14", "type": "l24", "path": B25_14, "adapters": [Q25_14], "keep": Q25_14C['keep'], "layers": Q25_14C['layers'], "w": 1.0, "bs": 8, "max_len": Q25_14C['max_len'], "twin": 0.35, "no_final_norm": True, "eos_id": Q25_14C['eos_id']},"""

    new_models_def = """  {"name": "ll8v5", "type": "l24", "path": BL, "adapters": ALL8V5, "keep": 24, "layers": L, "w": 1.8, "bs": 16, "max_len": 384, "twin": 0.35, "dp": True},
  {"name": "n14b", "type": "l24", "path": B14, "adapters": [N14B], "keep": N14BC['keep'], "layers": N14BC['layers'], "w": 2.5, "bs": 8, "max_len": N14BC['max_len'], "twin": 0.35, "no_final_norm": True, "eos_id": N14BC['eos_id']},
  {"name": "q25_14", "type": "l24", "path": B25_14, "adapters": [Q25_14], "keep": Q25_14C['keep'], "layers": Q25_14C['layers'], "w": 2.0, "bs": 8, "max_len": Q25_14C['max_len'], "twin": 0.35, "no_final_norm": True, "eos_id": Q25_14C['eos_id']},"""

    assert old_models_def in cell12_src, 'Old models definition not found in Cell 12!'
    cell12_src = cell12_src.replace(old_models_def, new_models_def)
    nb['cells'][12]['source'] = [cell12_src]

    # Save output notebook
    with open(out_nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print(f'Successfully generated {out_nb_path} with {len(nb["cells"])} cells.')

    # Create kernel-metadata.json
    metadata = {
        "id": "baonguyenduyquoc/ntl-champion-v61-glm4-sota-apex",
        "title": "NTL Champion V61 GLM4 SOTA Apex",
        "code_file": "champion-v61.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "keywords": [],
        "dataset_sources": [
            "seyitkaangunes/ntl-extra-api-passages",
            "abdullah1afify/ntl-generations",
            "dophuchung/ntl-proxy-generations",
            "zbigniewkirchner/ntl-qwen3-8b-lora-claude-head",
            "dhruvtibarewal/nlm-matched-clean",
            "josephadamski91/ntl-atlas-qwen3-8b-v1",
            "lucasboesen/ntl-free-proxy-generations",
            "dhruvtibarewal/nlm-enc-v5",
            "nishantdahal/ntl-handwriting-blend",
            "dhruvtibarewal/nlm-l24-adapters-v5",
            "prvsiyan/prvsiyans-extended-name-that-llm-dataset",
            "pjmathematician/ultrabert-380m-ntl",
            "ilyass323/llama-3-1-8b-bf16"
        ],
        "kernel_sources": [
            "lucasboesen/name-that-llm-glm4-9b-trunc25",
            "lucasboesen/name-that-llm-extract-encoders2",
            "lucasboesen/name-that-llm-electra-large-weights",
            "lucasboesen/name-that-llm-desklib-weights",
            "lucasboesen/name-that-llm-train-lora24-s7",
            "lucasboesen/name-that-llm-train-tokattn",
            "lucasboesen/name-that-llm-train-tokattn2",
            "vnton37nguyn/ntl-train-lora-qwen25-instyle"
        ],
        "competition_sources": [
            "name-that-llm"
        ],
        "model_sources": [
            "metaresearch/llama-3.1/Transformers/8b/2",
            "qwen-lm/qwen-3/Transformers/8b-base/1",
            "qwen-lm/qwen-3/Transformers/4b-base/1",
            "qwen-lm/qwen-3/Transformers/14b-base/1",
            "qwen-lm/qwen2.5/Transformers/14b/1",
            "qwen-lm/qwen2.5/Transformers/7b/1"
        ],
        "machine_shape": "NvidiaT4Dual"
    }

    meta_path = os.path.join(out_dir, 'kernel-metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f'Successfully generated {meta_path}.')

if __name__ == '__main__':
    build_v61()

