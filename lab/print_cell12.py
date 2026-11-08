import json

with open('champion_v61/champion-v61.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== CELL 11 & 12 ===")
print(''.join(nb['cells'][11]['source']))
print(''.join(nb['cells'][12]['source']))

