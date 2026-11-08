import json

with open('scratch/kchuang_0668/ntl-0668-stack-pool-prompt-rows.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [14, 16]:
    text = "".join(nb['cells'][idx]['source'])
    print(f"Cell {idx} total characters: {len(text)}")
    lines = text.split('\n')
    print(f"Cell {idx} total lines: {len(lines)}")
    print(f"First 5 lines: {lines[:5]}")
    print(f"Last 5 lines: {lines[-5:]}")

