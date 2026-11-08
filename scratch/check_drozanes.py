import json

with open('downloaded_kernels/drozanes_atlas_hungarian/name-that-llm-apex-ensemble-atlas-hungarian.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"drozanes cell count: {len(nb['cells'])}")
print("=== Cell 0 ===")
print(''.join(nb['cells'][0]['source'])[:1000])

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'MODELS =' in src or 'MEMBERS =' in src:
        print(f"=== Cell {i} (MODELS) ===")
        print(src[:1000])

