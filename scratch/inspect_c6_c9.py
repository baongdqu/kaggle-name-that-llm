import json

with open('champion_v30/champion-v30.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for c_idx in [6, 9]:
    print(f"=== Cell {c_idx} (len={len(nb['cells'][c_idx]['source'])}) ===")
    src = "".join(nb['cells'][c_idx]['source'])
    print(src[:1500])
    print("\n... [middle truncated] ...\n")
    print(src[-1500:])
