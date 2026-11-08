import json

with open('champion_v65/champion-v65.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('Num cells:', len(nb['cells']))
for i, c in enumerate(nb['cells']):
    ctype = c['cell_type']
    src = "".join(c.get('source', []))
    print(f"Cell {i} ({ctype}): length {len(src)} chars, first line: {src.splitlines()[0] if src else 'empty'}")

