import json

with open('champion_v30/champion-v30.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in V30: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    first_line = cell['source'][0] if cell['source'] else ''
    first_few = "".join(cell['source'][:3]).strip().replace('\n', ' ')
    print(f"Cell {i} ({cell['cell_type']}): {first_few[:120]}")
