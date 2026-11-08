import json

with open('champion_v44/champion-v44.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    lines = src.split('\n')
    header = lines[0][:70] if lines else 'empty'
    print(f"Cell {i} ({c['cell_type']}, {len(src)} chars): {header}")

