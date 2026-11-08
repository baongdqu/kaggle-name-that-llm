import json

with open('scratch/lightning_27b_spectral/qwen3-8-27b-lora-spectral-ensemble.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i in [0, 1, 2, 3, 4, 18, 21, 30, 39, 40, 41, 42]:
    if i < len(nb['cells']):
        c = nb['cells'][i]
        src = "".join(c['source'])
        print(f"=== Cell {i} ({c['cell_type']}) ===")
        print(src[:500])
        print("-" * 40)
