import json

with open('champion_v30/champion-v30.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for c_idx in [2, 11, 14]:
    print(f"=== Cell {c_idx} ===")
    print("".join(nb['cells'][c_idx]['source']))
