import json

with open('downloaded_kernels/alexchilton_0662/name-that-llm-0-662-all-public-what-failed.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Alex Chilton cell count: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    lines = src.split('\n')
    header = lines[0][:80] if lines else 'empty'
    ctype = cell.get('cell_type')
    print(f"Cell {i} ({ctype}): {header}")

