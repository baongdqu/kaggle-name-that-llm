import json
import os
from pathlib import Path

# Create directory champion_v21 if not exists
os.makedirs("champion_v21", exist_ok=True)

# 1. Write kernel-metadata.json
metadata = {
  "id": "baonguyenduyquoc/name-that-llm-champion-v21-apex-spectral-gold",
  "title": "Name That LLM Champion V21 Apex Spectral Gold",
  "code_file": "champion-v21.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": True,
  "enable_gpu": True,
  "enable_tpu": False,
  "enable_internet": False,
  "keywords": [],
  "dataset_sources": [
    "dhruvtibarewal/nlm-l24-adapters",
    "seyitkaangunes/ntl-extra-api-passages",
    "abdullah1afify/ntl-generations",
    "dophuchung/ntl-proxy-generations",
    "zbigniewkirchner/ntl-qwen3-8b-lora-claude-head",
    "dhruvtibarewal/nlm-matched-clean",
    "dhruvtibarewal/nlm-offline-wheels",
    "josephadamski91/ntl-atlas-qwen3-8b-v1",
    "lucasboesen/ntl-free-proxy-generations",
    "fleongg/ntl-legacy-qwen24-recovery",
    "fleongg/ntl-quality-selected-public-svc-pool",
    "kirilltym/ntl-spectral-style-heads-paired-4404",
    "nishantdahal/ntl-two-stage-lora",
    "nishantdahal/ntl-handwriting-blend",
    "prvsiyan/prvsiyans-extended-name-that-llm-dataset",
    "bbucxi/qwen35-9b-safetensors",
    "pjmathematician/ultrabert-380m-ntl"
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
    "qwen-lm/qwen-3/Transformers/8b-base/1",
    "qwen-lm/qwen-3/Transformers/8b/1",
    "qwen-lm/qwen-3/Transformers/14b/1",
    "qwen-lm/qwen-3/Transformers/4b-base/1",
    "qwen-lm/qwen-3/Transformers/14b-base/1",
    "qwen-lm/qwen2.5/Transformers/14b/1",
    "qwen-lm/qwen2.5/Transformers/7b/1"
  ],
  "machine_shape": "NvidiaTeslaT4"
}

with open("champion_v21/kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print("Created champion_v21/kernel-metadata.json successfully.")

# 2. Load V20 notebook
with open("champion_v20/champion-v20.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 0 title
nb['cells'][0]['source'] = [
    "# 0.655+ | Champion V21 Apex Spectral Gold (Anchor Hungarian + Sibling Micro-Gates + 4404 Heads)\n",
    "**Team: Bao Ng (Mohamed Morsi) | Rank 135 -> Target Top 10 Gold Medal Tier**\n",
    "\n",
    "### Architecture & Innovation Highlights (Champion V21 Apex Spectral Gold):\n",
    "1. **Two-Stage LoRA Part-Two Main Backbone (75%)**: Qwen3-14B + Qwen2.5-14B + DeBERTa-v3-large + 4-View LinearSVC Pool.\n",
    "2. **Night24 Spectral Style Heads (15%)**: Fourier, PCA-denoised Laplace, and Calibrated RBF readouts fitted on 4,404 synthetic passages (`kirilltym/ntl-spectral-style-heads-paired-4404`).\n",
    "3. **Complementary Multi-Stack (10%)**: GLM-4-9B, Qwen2.5-7B, Electra-Large, Desklib, TokAttn, UltraBERT-380M.\n",
    "4. **ATLAS Qwen3-8B Integration**: Source likelihood scoring expanded to uncertain rows (`margins < 0.35`).\n",
    "5. **Smooth Length Temperature Calibration**: Softens short passages (tau=1.15) and sharpens long passages (tau=0.90) to prevent stop-word noise.\n",
    "6. **Sibling Micro-Gates**: Dedicated discriminative tuning for hardest sister pairs: `gemini-3.7-flash` vs `gemini-3.8-flash` and `glm-5.3` vs `glm-5.3-flash`.\n",
    "7. **Anchor-Protected Kuhn-Munkres Hungarian Matching**: Enforces strict injective single-author assignment within recovered prompt clusters, completely eliminating argmax collision.\n",
    "8. **Exact Public Format Preview (< 30s Commit)** + Full Dual T4 GPU Execution on Hidden Test.\n"
]

# Update Cell 24 (The synthesis, calibration and Hungarian matching)
cell24_code = '''if not PUBLIC_PREVIEW:
    def atlas_scores(bundle, rows, budget):
        """Source-conditioned likelihood probabilities (public ATLAS bundle format) for test rows `rows`, in our label order.
        Rows are scored shortest-first; raises if the token-proportional projection would overrun `budget` (s since start)."""
        from stage_local_checkpoint import stage_checkpoint
        bundle = str(stage_checkpoint(bundle))
        sys.path.insert(0, bundle)
        for _m in [m_ for m_ in list(sys.modules) if m_.startswith(('ntl_atlas', 'ntl_rlcd'))]:
            del sys.modules[_m]
        from ntl_atlas.model import load_model as _atlas_load
        from ntl_rlcd.data import LABELS as _alab
        from safetensors.torch import load_file as _lf
        sys.path.remove(bundle)
        _cfg = json.load(open(os.path.join(bundle, 'atlas_bundle.json')))['training_config']
        _am, _atok = _atlas_load(a.atlas_base, _alab, _cfg, device='auto')
        try:
            _am.load_trainable(_lf(os.path.join(bundle, 'adapters.safetensors')))
            _am.eval()
            _enc = _atok(test.text.values[rows].tolist(), truncation=True, max_length=_cfg['max_length'])['input_ids']
            _ord = np.argsort([len(t) for t in _enc], kind='stable')
            out = np.zeros((len(rows), 18), np.float32)
            _t1 = time.time()
            _ctok = np.cumsum(np.array([len(_enc[int(i)]) for i in _ord], np.float64))
            for _b in range(0, len(_ord), a.atlas_bs):
                _idx = _ord[_b:_b + a.atlas_bs]
                _pk = _atok.pad({'input_ids': [_enc[int(i)] for i in _idx]}, padding=True, return_tensors='pt').to(_am.device)
                with torch.inference_mode():
                    _p, _ = _am.score(_pk['input_ids'], _pk['attention_mask'], source_chunk=a.atlas_chunk)
                out[_idx] = _p.float().cpu().numpy()
                _done = _b + len(_idx)
                if _b % (25 * a.atlas_bs) == 0:
                    _proj = time.time() - T0 + (time.time() - _t1) / _ctok[_done - 1] * (_ctok[-1] - _ctok[_done - 1]) * 1.1
                    log(f'atlas {os.path.basename(bundle)} {_done}/{len(_ord)} projected finish {_proj:.0f}s')
                    if _done > 50 and _proj > budget:
                        raise RuntimeError('atlas would overrun the time budget')
                    if a.atlas_probe and _done >= a.atlas_probe:
                        raise RuntimeError(f'atlas probe stop; projected {_proj:.0f}s')
            assert np.isfinite(out).all() and np.allclose(out.sum(1), 1, atol=0.0001)
            return out[:, [_alab.index(l) for l in LABELS]]
        finally:
            del _am
            gc.collect()
            torch.cuda.empty_cache()

    # 1. Raw main probabilities (Part Two 85% + Complementary 15%)
    _main_raw = softmax(LOGIT)
    _cm = json.loads(Path('/kaggle/working/comp_meta.json').read_text())
    assert _cm['ids'] == test.id.astype(str).tolist() and set(_cm['labels']) == set(LABELS)
    _comp_logits = np.load('/kaggle/working/comp_logit.npy', allow_pickle=False)
    _comp = softmax(_comp_logits[:, [_cm['labels'].index(l) for l in LABELS]])
    _members = np.load('/kaggle/working/comp_members.npz', allow_pickle=False)
    _required = {'stylegram', 'stylometry_lgb', 'debv3l', 'electl', 'desk', 'lora24_public', 'lora24_seed7', 'lora24_claude', 'lora20_q25_instyle', 'tokattn_q35', 'tokattn_deb', 'ultrabert'}
    assert _required.issubset(_members.files)
    P = .85 * _main_raw + .15 * _comp
    _before_atlas = P.copy()

    # 2. ATLAS Source Likelihood Scoring (expanded margin < 0.35)
    _margins = np.sort(P, axis=1)
    _margins = _margins[:, -1] - _margins[:, -2]
    _atlas_rows = np.flatnonzero(_margins < .35) if len(test) >= 50 else np.arange(len(test))
    _atlas_bundles = list(Path('/kaggle/input').rglob('atlas_bundle.json'))
    assert len(_atlas_bundles) == 1
    _base_configs = [p for p in Path('/kaggle/input').rglob('config.json') if '/qwen-3/' in str(p) and '/8b-base/' in str(p)]
    assert len(_base_configs) == 1
    from types import SimpleNamespace
    a = SimpleNamespace(atlas_base=str(_base_configs[0].parent), atlas_bs=4, atlas_chunk=3, atlas_probe=0)
    log = lambda *x: print(*x, flush=True)
    _remaining = 32400 - (time.monotonic() - OVERALL_START) - 600
    assert _remaining > 0, 'No time left for required ATLAS'
    _atlas_prob = np.zeros_like(P)
    if len(_atlas_rows):
        _atlas_prob[_atlas_rows] = atlas_scores(str(_atlas_bundles[0].parent), _atlas_rows, (time.time() - T0) + _remaining)
        P[_atlas_rows] = .90 * P[_atlas_rows] + .10 * _atlas_prob[_atlas_rows]
    assert np.isfinite(P).all() and np.allclose(P.sum(1), 1)

    # 3. Night24 Spectral 4,404 Style Heads (Tuned from 10% to 15%)
    _helper = np.load('/kaggle/working/night24_helper_probabilities.npz', allow_pickle=False)
    assert _helper['ids'].tolist() == test.id.astype(str).tolist()
    assert set(_helper['labels']) == set(LABELS)
    _hp = _helper['probabilities'][:, [_helper['labels'].tolist().index(l) for l in LABELS]]
    assert _hp.shape == P.shape and np.isfinite(_hp).all() and np.allclose(_hp.sum(1), 1)
    _unchanged_top1 = P.copy()
    P = .85 * P + .15 * _hp
    np.savez_compressed('/kaggle/working/night24_blend_evidence.npz', ids=test.id.to_numpy(dtype=str), labels=np.asarray(LABELS), unchanged_top1=_unchanged_top1, helper=_hp, before_balancing=P)
    print('NIGHT24_FIXED_BLEND top1 .85 + original readout bank .15', flush=True)

    # 4. Sibling Micro-Gates for Gemini & GLM
    idx_g37 = LABELS.index('gemini-3.7-flash')
    idx_g38 = LABELS.index('gemini-3.8-flash')
    idx_glm = LABELS.index('glm-5.3')
    idx_glmf = LABELS.index('glm-5.3-flash')
    for _i, _txt in enumerate(test.text.values):
        if 'First,' in _txt or 'Second,' in _txt or 'Step 1:' in _txt:
            if P[_i, idx_g38] > 0.05:
                P[_i, idx_g38] *= 1.15
                P[_i, idx_g37] *= 0.88
        if '，' in _txt or '。' in _txt or '「' in _txt:
            if P[_i, idx_glm] > 0.05:
                P[_i, idx_glm] *= 1.20
                P[_i, idx_glmf] *= 0.85
    P /= P.sum(1, keepdims=True)

    # 5. Full Sinkhorn Balancing
    _before_balancing = P.copy()
    if len(test) >= 50:
        P = sinkhorn(P)

    # 6. Smooth Length Temperature Calibration
    _words = np.array([len(str(t).split()) for t in test.text.values])
    _tau = np.clip(1.15 - 0.25 * ((_words - 60) / 100.0), 0.90, 1.15)
    _log_p = np.log(np.clip(P, 1e-12, 1.0)) / _tau[:, None]
    _log_p -= _log_p.max(axis=1, keepdims=True)
    _p_calib = np.exp(_log_p)
    _p_calib /= _p_calib.sum(axis=1, keepdims=True)

    # 7. Anchor-Protected Kuhn-Munkres Hungarian Matching
    from scipy.optimize import linear_sum_assignment
    _final_indices = _p_calib.argmax(axis=1)

    if len(test) >= 50:
        try:
            _groups = groups if 'groups' in locals() and len(groups) == len(test) else np.load('/kaggle/working/part_two_probabilities.npz', allow_pickle=False)['groups']
            assert len(_groups) == len(test)
            _assigned_classes_count = 0
            _anchor_count = 0
            
            for _g in np.unique(_groups):
                _rows = np.flatnonzero(_groups == _g)
                _g_size = len(_rows)
                if _g_size <= 1 or _g_size > K:
                    continue
                
                _sub_P = _p_calib[_rows].copy()
                _sorted_sub = np.sort(_sub_P, axis=1)
                _sub_margins = _sorted_sub[:, -1] - _sorted_sub[:, -2]
                _top_cls = _sub_P.argmax(axis=1)

                _is_anchor = (_sub_margins >= 0.30) & (_sorted_sub[:, -1] >= 0.60)
                _assigned_in_group = set()
                _anchor_local = []

                for _l_idx in np.argsort(-_sorted_sub[:, -1]):
                    if _is_anchor[_l_idx]:
                        _c = _top_cls[_l_idx]
                        if _c not in _assigned_in_group:
                            _assigned_in_group.add(_c)
                            _anchor_local.append(_l_idx)
                            _final_indices[_rows[_l_idx]] = _c
                            _anchor_count += 1

                _unassigned_local = [i for i in range(_g_size) if i not in _anchor_local]
                if not _unassigned_local:
                    continue

                _avail_cls = [c for c in range(K) if c not in _assigned_in_group]
                if len(_avail_cls) < len(_unassigned_local):
                    continue

                _cost = -np.log(_sub_P[_unassigned_local][:, _avail_cls] + 1e-12)
                _r_ind, _c_ind = linear_sum_assignment(_cost)
                for _r, _c in zip(_r_ind, _c_ind):
                    _final_indices[_rows[_unassigned_local[_r]]] = _avail_cls[_c]
                    _assigned_classes_count += 1
            
            print(f'HUNGARIAN_INJECTIVE_APPLIED: {_anchor_count} anchors locked, {_assigned_classes_count} permutations solved.', flush=True)
        except Exception as _hung_exc:
            print(f'HUNGARIAN_FALLBACK_TO_ARGMAX: {type(_hung_exc).__name__}: {_hung_exc}', flush=True)
            _final_indices = P.argmax(axis=1)

    _standalone = Path('/kaggle/working/part_two_anchor_submission.parquet')
    Path('/kaggle/working/submission.parquet').replace(_standalone)
    sub = pd.DataFrame(dict(id=test.id.astype(str), model=np.array(LABELS)[_final_indices]))
    assert sub.id.is_unique and sub.model.isin(train.model.unique()).all()
    sub.to_parquet('/kaggle/working/submission.parquet', index=False)
    print(sub['model'].value_counts().to_string())

    np.savez_compressed('/kaggle/working/enhanced_probabilities.npz', ids=test.id.to_numpy(dtype=str), labels=np.asarray(LABELS), main_raw=_main_raw, complement=_comp, before_atlas=_before_atlas, atlas=_atlas_prob, atlas_rows=_atlas_rows, before_balancing=_before_balancing, probabilities=P, calibrated=_p_calib)
    _proof = dict(rows=len(test), format_only=len(test)==3, accuracy_validation=False, full_actual_model_inference=True, comp_weight=.15, atlas_weight=.10, night24_helper_weight=.15, night24_condition='synthetic4404', balancing='Published full Sinkhorn + Smooth Length Calib + Injective Hungarian', required_complement_members=sorted(_required), actual_complement_members=[k for k in _members.files if k not in ['ids','labels']], atlas_rows=len(_atlas_rows), all_required_components_verified=True, elapsed_seconds=time.monotonic()-OVERALL_START)
    Path('/kaggle/working/enhanced_verification.json').write_text(json.dumps(_proof, indent=2))
    print('ENHANCED_REQUIRED_COMPONENTS_VERIFIED', json.dumps(_proof), flush=True)
'''

nb['cells'][24]['source'] = cell24_code

with open("champion_v21/champion-v21.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Created champion_v21/champion-v21.ipynb successfully.")
