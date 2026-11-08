import json, os, copy

# 1. Read metadata from V68
meta = json.load(open('champion_v68/kernel-metadata.json', 'r', encoding='utf-8'))
meta['id'] = 'baonguyenduyquoc/ntl-champion-v75-sibling-lora-apex'
meta['title'] = 'NTL Champion V75 Sibling LoRA Apex'
meta['code_file'] = 'champion-v75.ipynb'

# Add our newly created dataset
ds = 'baonguyenduyquoc/ntl-sibling-contrastive-lora24-weights'
if ds not in meta['dataset_sources']:
    meta['dataset_sources'].append(ds)

with open('champion_v75/kernel-metadata.json', 'w', encoding='utf-8') as mf:
    json.dump(meta, mf, indent=2)

# 2. Read notebook from V68
nb = json.load(open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8'))

# Update Title cell 0
nb['cells'][0]['source'] = [
    '# Name That LLM Champion V75 Sibling-Contrastive LoRA Apex\n',
    '\n',
    '### Core Paradigm Shift (0.662 Golden Anchor + In-Model Sibling Separation):\n',
    '- **Sibling-Aware Multi-Similarity Contrastive LoRA L24 (Qwen3-8B-Base)**: Trained with $W_{ij}=3.0$ on sibling clusters (Gemini Flash, GLM, GPT-5).\n',
    '- **5-Fold OOF Balanced Acc = 47.24%** (+4.74% over vanilla LoRA baseline 42.5%).\n',
    '- **Multi-Layer Representation**: 32,768-D features (8 layers x 4096) with Dual-Readout Linear Heads (plain + centered).\n',
    '- **Integrated into Golden Anchor V68**: Combined with Latin-Square Hungarian Matching (win-rate 98%) & Sinkhorn Damped 0.45.\n'
]

# In Cell 2 (Setup & Discovery), discover our sibling LoRA checkpoint
cell2 = ''.join(nb['cells'][2]['source'])
sibling_discovery = """
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
cell2_updated = cell2 + sibling_discovery
nb['cells'][2]['source'] = [line + '\n' for line in cell2_updated.splitlines()]

# In Cell 12 (Models & Execution), add sibling LoRA to MODELS list with defensive fallback
cell12 = ''.join(nb['cells'][12]['source'])
target = 'MODELS = ['
idx = cell12.find(target)
if idx != -1:
    idx_end = idx + len(target)
    sibling_member = """
  {"name": "q8_sibling_l24", "type": "l24", "path": B8, "adapters": [SIBLING_LORA], "keep": SIBLING_LORAC['keep'], "layers": SIBLING_LORAC['layers'], "w": 1.2, "bs": 16, "max_len": SIBLING_LORAC['max_len'], "twin": 0.35, "dp": True},"""
    cell12_updated = cell12[:idx_end] + sibling_member + cell12[idx_end:]
    nb['cells'][12]['source'] = [line + '\n' for line in cell12_updated.splitlines()]

with open('champion_v75/champion-v75.ipynb', 'w', encoding='utf-8') as nbf:
    json.dump(nb, nbf, indent=1)

print('Successfully created Champion V75 notebook and metadata in champion_v75/!')

