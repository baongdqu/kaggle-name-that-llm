import json

with open('scratch/kchuang_0668/ntl-0668-stack-pool-prompt-rows.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== CELL 11 v34_pairwise.py ===")
src = "".join(nb['cells'][11]['source'])
for line in src.split('\n')[:40]:
    print(line)

print("=== CELL 12 Invocation ===")
src12 = "".join(nb['cells'][12]['source'])
for line in src12.split('\n'):
    print(line)

