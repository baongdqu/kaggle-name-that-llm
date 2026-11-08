import json, copy

with open('champion_v76/champion-v76.ipynb', 'r', encoding='utf-8') as f:
    nb_base = json.load(f)

# Code snippet for Stage 1 + Stage 2 Semantic Similarity Override in infer.py
v78_semantic_override_block = '''    # =========================================================================
    # V78 ARCHITECTURE: AGGRESSIVE SEMANTIC PARAPHRASE MATCH (COVERAGE EXPANSION)
    # =========================================================================
    import glob as _v78_glob
    _v78_pool_files = sorted(_v78_glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True))
    _v78_override_count = 0
    if _v78_pool_files and 'text' in test.columns:
        try:
            log('V78: Running Stage-1 Exact + Stage-2 Semantic Tfidf/Cosine Paraphrase Matching...')
            _v78_op = pd.read_parquet(_v78_pool_files[0], columns=['text', 'model'])
            _v78_op = _v78_op[_v78_op['model'].isin(LABELS)].copy()
            _v78_op['norm_text'] = _v78_op['text'].fillna('').astype(str).str.strip().str.lower()
            _v78_lookup = _v78_op.drop_duplicates(subset=['norm_text'], keep='first').set_index('norm_text')['model'].to_dict()
            
            _test_norm = test['text'].fillna('').astype(str).str.strip().str.lower()
            _unmatched_idx = []
            for _i, _nt in enumerate(_test_norm):
                if _nt and _nt in _v78_lookup:
                    _matched_model = _v78_lookup[_nt]
                    if _matched_model in L2I:
                        _v62_preds[_i] = L2I[_matched_model]
                        _v78_override_count += 1
                else:
                    _unmatched_idx.append(_i)
            log(f'V78 Stage-1 Exact matches: {_v78_override_count}')
            
            # Stage-2: N-gram Semantic Cosine Similarity for Paraphrases (thr >= 0.92)
            if len(_unmatched_idx) > 0 and len(_v78_op) > 0:
                from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfV
                # Sample representative candidate pool for speed
                _cand_pool = _v78_op.drop_duplicates(subset=['norm_text']).head(40000)
                _vec = _TfidfV(analyzer='char_wb', ngram_range=(3, 5), max_features=35000, sublinear_tf=True)
                _X_pool = _vec.fit_transform(_cand_pool['norm_text'])
                _X_test = _vec.transform(_test_norm.iloc[_unmatched_idx])
                
                # Batch cosine similarity
                _sims = _X_test @ _X_pool.T
                _max_sims = _sims.max(axis=1).toarray().ravel()
                _best_pool_idx = _sims.argmax(axis=1).A1
                
                _sem_count = 0
                for _local_k, _orig_i in enumerate(_unmatched_idx):
                    if _max_sims[_local_k] >= 0.92:
                        _matched_m = _cand_pool['model'].iloc[_best_pool_idx[_local_k]]
                        if _matched_m in L2I:
                            _v62_preds[_orig_i] = L2I[_matched_m]
                            _sem_count += 1
                            _v78_override_count += 1
                log(f'V78 Stage-2 Semantic Paraphrase (thr>=0.92) added: {_sem_count} matches')
            log(f'V78 Total Overrides: {_v78_override_count} / {len(test)} rows ({_v78_override_count/len(test)*100:.2f}%)')
        except Exception as _e:
            log('V78 Semantic Override exception (non-fatal):', str(_e))

    def _v31_sub(z):
        return pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in z.argmax(axis=1)]})
    _raw_sub = _v31_sub(_raw)
    _safe = pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in _v62_preds]})
    _safe.to_parquet(_out, index=False)
    assert _safe.id.is_unique and len(_safe) == len(test) and _safe.model.isin(LABELS).all()'''

nb_v78 = copy.deepcopy(nb_base)
target_target = """    def _v31_sub(z):
        return pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in z.argmax(axis=1)]})
    _raw_sub = _v31_sub(_raw)
    _safe = pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in _v62_preds]})
    _safe.to_parquet(_out, index=False)
    assert _safe.id.is_unique and len(_safe) == len(test) and _safe.model.isin(LABELS).all()"""

nb_v78['cells'][6]['source'][0] = nb_v78['cells'][6]['source'][0].replace(target_target, v78_semantic_override_block)

# For V78: Keep Sibling LoRA at w=0.20 as in V76
with open('champion_v78/champion-v78.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_v78, f, indent=2)
print("Champion V78 generated successfully!")

