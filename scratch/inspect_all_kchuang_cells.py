import json

with open('scratch/kchuang_0668/ntl-0668-stack-pool-prompt-rows.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("KCHUANG ALL CODE CELLS:")
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        first = c['source'][0].strip() if c['source'] else ""
        print(f"Cell {i:2d} ({len(c['source'])} lines): {first[:70]}")

