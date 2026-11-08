import json

with open('downloaded_kernels/lightningv08_0661/0-661-qwen3-8-27b-lora-spectral-ensemble.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Lightningv08 cell count: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    lines = src.split('\n')
    header = lines[0][:80] if lines else 'empty'
    ctype = cell.get('cell_type')
    print(f"Cell {i} ({ctype}): {header}")
