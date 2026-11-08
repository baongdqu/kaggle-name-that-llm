import json, os

# 1. Metadata for V76
meta = json.load(open('champion_v68/kernel-metadata.json', 'r', encoding='utf-8'))
meta['id'] = 'baonguyenduyquoc/ntl-champion-v76-open-pool-apex'
meta['title'] = 'NTL Champion V76 Open Pool Apex'
meta['code_file'] = 'champion-v76.ipynb'

# Attach crtproduction/name-that-llm-open-pool (211k passages)
if 'crtproduction/name-that-llm-open-pool' not in meta['dataset_sources']:
    meta['dataset_sources'].append('crtproduction/name-that-llm-open-pool')

# Attach our sibling LoRA weights as well (safe)
if 'baonguyenduyquoc/ntl-sibling-contrastive-lora24-weights' not in meta['dataset_sources']:
    meta['dataset_sources'].append('baonguyenduyquoc/ntl-sibling-contrastive-lora24-weights')

with open('champion_v76/kernel-metadata.json', 'w', encoding='utf-8') as mf:
    json.dump(meta, mf, indent=2)

# 2. Notebook for V76
nb = json.load(open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8'))

# Update Title cell 0
nb['cells'][0]['source'] = [
    '# Name That LLM Champion V76 Open Pool & Stylometric Apex\n',
    '\n',
    '### Core Paradigm Shift (0.662 Golden Anchor + Clean Open Pool 198k + Safe Sibling LoRA):\n',
    '- **Clean Open Pool Integration (crtproduction/name-that-llm-open-pool)**: Upgrading TF-IDF, Distort and 4-View SVC with 198,000 clean passage pool (reproducing the 0.668 breakthrough mechanism).\n',
    '- **Safe Sibling-Aware Contrastive LoRA L24**: Calibrated at gentle orthogonal weight w=0.20 to protect Golden Anchor precision.\n',
    '- **Latin-Square Kuhn-Munkres Bipartite Matching**: Retains 100% win-rate Hungarian matching (size >= 5) & Damped Sinkhorn 0.45.\n'
]

# Cell 2: Add Open Pool discovery & Sibling LoRA discovery
cell2 = ''.join(nb['cells'][2]['source'])
open_pool_code = """
# Clean Open Pool (Variant E from 0.668 public breakthrough)
import glob as _glob
import pandas as _pd
_pool_src = sorted(_glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True))
if _pool_src:
    _op = _pd.read_parquet(_pool_src[0])
    _lab18 = set(_pd.read_csv(DATA + '/train.csv').model.unique())
    _op = _op[_op['model'].isin(_lab18)].copy()
    if 'flag_train_prompt_regen' in _op.columns:
        _op = _op[~_op['flag_train_prompt_regen'].astype(bool)]
    for _f in ('flag_gemini37_routing_risk',):
        if _f in _op.columns:
            _n = int(_op[_f].astype(bool).sum()); _op = _op[~_op[_f].astype(bool)]
            print('Clean Open Pool dropped', _f, _n, flush=True)
    _op = _op[['text', 'model']].copy(); _op['text'] = _op['text'].fillna('').astype(str)
    _op.to_parquet('/kaggle/working/pool_open_safe.parquet', index=False)
    POOL = '/kaggle/working/pool_open_safe.parquet'
    print('Clean Open Pool activated for SVC pool with rows:', len(_op), flush=True)
else:
    print('Clean Open Pool not found; keeping default nlm_pool_v5', flush=True)

# Sibling Contrastive LoRA L24 discovery
def find_sibling_lora():
    meta = glob.glob('/kaggle/input/**/ntl-sibling-contrastive-lora24-weights/**/meta.json', recursive=True)
    if not meta:
        meta = glob.glob('/kaggle/input/**/meta.json', recursive=True)
        meta = [m for m in meta if 'sibling' in m.lower()]
    if meta:
        return os.path.dirname(meta[0]), json.load(open(meta[0]))['cfg']
    return None, {'keep': 24, 'layers': [2, 4, 6, 8, 12, 16, 20, 24], 'max_len': 384, 'eos_id': None}

SIBLING_LORA, SIBLING_LORAC = find_sibling_lora()
print('Sibling Contrastive LoRA discovered at:', SIBLING_LORA)
"""
cell2_updated = cell2 + open_pool_code
nb['cells'][2]['source'] = [line + '\n' for line in cell2_updated.splitlines()]

# Cell 12: Add Sibling LoRA with gentle weight w=0.20
cell12 = ''.join(nb['cells'][12]['source'])
target = 'MODELS = ['
idx = cell12.find(target)
if idx != -1:
    idx_end = idx + len(target)
    sibling_member = """
  {"name": "q8_sibling_l24", "type": "l24", "path": B8, "adapters": [SIBLING_LORA], "keep": SIBLING_LORAC['keep'], "layers": SIBLING_LORAC['layers'], "w": 0.20, "bs": 16, "max_len": SIBLING_LORAC['max_len'], "twin": 0.35, "dp": True},"""
    cell12_updated = cell12[:idx_end] + sibling_member + cell12[idx_end:]
    nb['cells'][12]['source'] = [line + '\n' for line in cell12_updated.splitlines()]

with open('champion_v76/champion-v76.ipynb', 'w', encoding='utf-8') as nbf:
    json.dump(nb, nbf, indent=1)

print('Champion V76 files successfully created in champion_v76/!')

