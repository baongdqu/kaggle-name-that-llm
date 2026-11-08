import json

with open('scratch/lucas_desklib/name-that-llm-desklib-weights.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Cells in desklib:", len(nb['cells']))
for i, c in enumerate(nb['cells']):
    src = "".join(c.get('source', []))
    print(f"--- Cell {i} ---")
    print(src[:600])

