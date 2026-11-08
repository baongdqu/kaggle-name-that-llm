import json

with open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in V68: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src = "".join(c.get('source', []))
    first_line = src.split('\n')[0] if src else ""
    print(f"Cell {i} ({c['cell_type']}) [{len(src)} chars]: {first_line[:80]}")

