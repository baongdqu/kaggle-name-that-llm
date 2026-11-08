import json, copy

with open('champion_v76/champion-v76.ipynb', 'r', encoding='utf-8') as f:
    nb_base = json.load(f)

# Code snippet for Stage 1 Exact Match Pool Override in infer.py
v77_exact_override_block = '''    # =========================================================================
    # V77 ARCHITECTURE: STAGE-1 CLEAN OPEN POOL EXACT OVERRIDE + ANCHORED OT
    # =========================================================================
    import glob as _v77_glob
    _v77_pool_files = sorted(_v77_glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True))
    _v77_override_count = 0
    if _v77_pool_files and 'text' in test.columns:
        try:
            log('V77: Scanning Clean Open Pool for Exact Text Matches...')
            _v77_op = pd.read_parquet(_v77_pool_files[0], columns=['text', 'model'])
            _v77_op = _v77_op[_v77_op['model'].isin(LABELS)].copy()
            _v77_op['norm_text'] = _v77_op['text'].fillna('').astype(str).str.strip().str.lower()
            # drop duplicates in pool to avoid contradictory overrides
            _v77_lookup = _v77_op.drop_duplicates(subset=['norm_text'], keep='first').set_index('norm_text')['model'].to_dict()
            
            _test_norm = test['text'].fillna('').astype(str).str.strip().str.lower()
            for _i, _nt in enumerate(_test_norm):
                if _nt and _nt in _v77_lookup:
                    _matched_model = _v77_lookup[_nt]
                    if _matched_model in L2I:
                        _v62_preds[_i] = L2I[_matched_model]
                        _v77_override_count += 1
            log(f'V77 Stage-1 Exact Pool Override applied to {_v77_override_count} / {len(test)} rows ({_v77_override_count/len(test)*100:.2f}%)')
        except Exception as _e:
            log('V77 Exact Pool Override exception (non-fatal, falling back to Anchor):', str(_e))

    def _v31_sub(z):
        return pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in z.argmax(axis=1)]})
    _raw_sub = _v31_sub(_raw)
    _safe = pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in _v62_preds]})
    _safe.to_parquet(_out, index=False)
    assert _safe.id.is_unique and len(_safe) == len(test) and _safe.model.isin(LABELS).all()'''

# For V77: Sibling LoRA is w=0.0 in MODELS (Cell 12)
nb_v77 = copy.deepcopy(nb_base)
target_target = """    def _v31_sub(z):
        return pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in z.argmax(axis=1)]})
    _raw_sub = _v31_sub(_raw)
    _safe = pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in _v62_preds]})
    _safe.to_parquet(_out, index=False)
    assert _safe.id.is_unique and len(_safe) == len(test) and _safe.model.isin(LABELS).all()"""

nb_v77['cells'][6]['source'][0] = nb_v77['cells'][6]['source'][0].replace(target_target, v77_exact_override_block)

# In Cell 12: Set w=0.0 for q8_sibling_l24 (or remove from blend)
cell12_src = "".join(nb_v77['cells'][12]['source'])
cell12_src = cell12_src.replace('"w": 0.20', '"w": 0.0')
nb_v77['cells'][12]['source'] = [cell12_src]

with open('champion_v77/champion-v77.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_v77, f, indent=2)
print("Champion V77 generated successfully!")

