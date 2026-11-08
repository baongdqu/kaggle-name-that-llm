import json

with open('champion_v47/champion-v47.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Inspect cell 10
print("=== CELL 10 ===")
print(''.join(nb['cells'][10]['source']))

# Inspect cell 12
print("=== CELL 12 ===")
print(''.join(nb['cells'][12]['source']))

