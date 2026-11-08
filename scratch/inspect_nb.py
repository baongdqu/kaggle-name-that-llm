import json

with open('downloaded_kernels/omerfarukmerey_0663/0-663-0-660-v31-atlas-20.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Cell count: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    lines = src.split('\n')
    header = lines[0][:80] if lines else 'empty'
    ctype = cell.get('cell_type')
    print(f"Cell {i} ({ctype}): {header}")
