import json

with open('scratch/kchuang_0668/ntl-0668-stack-pool-prompt-rows.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [7, 9, 11]:
    text = "".join(nb['cells'][idx]['source'])
    lines = text.split('\n')
    print(f"=== Cell {idx} ({len(lines)} lines) ===")
    print(lines[0])
    print(lines[1] if len(lines) > 1 else "")

