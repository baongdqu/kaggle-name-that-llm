import json

with open('train_sovereign_modernbert_31k/train-sovereign-modernbert-31k.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Cells count:", len(nb['cells']))
for i, c in enumerate(nb['cells']):
    src = "".join(c.get('source', []))
    print(f"--- Cell {i} ({c['cell_type']}) [{len(src)} chars] ---")
    print(src[:400])
    print()

