import json

with open('champion_v44/champion-v44.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== CELL 2 of V44 ===")
print(''.join(nb['cells'][2]['source']))

print("=== CELL 11 of V44 ===")
print(''.join(nb['cells'][11]['source']))

