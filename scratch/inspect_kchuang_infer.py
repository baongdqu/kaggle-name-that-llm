import json

with open('scratch/kchuang_0668/ntl-0668-stack-pool-prompt-rows.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in [14, 16]:
    c = nb['cells'][i]
    src = "".join(c['source'])
    print(f"=== CELL {i} ===")
    for line in src.split('\n')[:35]:
        print(line)

