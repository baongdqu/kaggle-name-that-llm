import json

with open('scratch/kozykappa/name-that-llm-enhanced-0-651-recipe.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = "".join(c['source'])
    print(f"=== Cell {i} ({c['cell_type']}) ===")
    print(src[:400])
    print("-" * 40)
