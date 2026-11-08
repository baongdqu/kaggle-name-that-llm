import json

with open('scratch/adamski_atlas/teach-the-model-10-atlas.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    print(f"\n{'='*20} CELL {i} ({c['cell_type']}) {'='*20}")
    src = "".join(c.get('source', ''))
    print(src)
