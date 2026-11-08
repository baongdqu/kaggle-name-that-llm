import json, re

# Load champion-v76 as base
with open('champion_v76/champion-v76.ipynb', 'r', encoding='utf-8') as f:
    nb_base = json.load(f)

# Inspect exact lines in cell 6 around submission writing
infer_src = nb_base['cells'][6]['source'][0]
target_block = """    def _v31_sub(z):
        return pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in z.argmax(axis=1)]})
    _raw_sub = _v31_sub(_raw)
    _safe = pd.DataFrame({'id': test.id.values, 'model': [LABELS[int(i)] for i in _v62_preds]})
    _safe.to_parquet(_out, index=False)
    assert _safe.id.is_unique and len(_safe) == len(test) and _safe.model.isin(LABELS).all()"""

print("Target block in infer_src:", target_block in infer_src)

