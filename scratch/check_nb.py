import json

with open('champion_v63/champion-v63.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    lines = src.strip().split('\n')
    fl = lines[0] if lines else 'EMPTY'
    print(f"Cell {i} ({c.get('cell_type')}): {fl[:80]} (len={len(src)})")

