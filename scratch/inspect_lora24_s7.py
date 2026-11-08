import json

with open('scratch/lucas_lora24_s7/name-that-llm-train-lora24-s7.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Cells in lora24_s7:", len(nb['cells']))
for i, c in enumerate(nb['cells']):
    src = "".join(c.get('source', []))
    print(f"--- Cell {i} ---")
    print(src[:600])

