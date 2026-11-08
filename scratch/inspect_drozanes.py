import json

with open('downloaded_kernels/drozanes_apex_ensemble/name-that-llm-apex-ensemble-atlas-hungarian.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"drozanes cell count: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    lines = src.split('\n')
    header = lines[0][:80] if lines else 'empty'
    print(f"Cell {i} ({cell.get('cell_type')}): {header}")
