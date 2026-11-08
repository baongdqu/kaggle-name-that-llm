import json

with open('scratch/lucas_enc2/name-that-llm-extract-encoders2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = "".join(c.get('source', []))
    print(f"--- Cell {i} ---")
    print(src[:800])

