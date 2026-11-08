import json

with open('champion_v20/champion-v20.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== CELL 13 ===")
print("".join(nb['cells'][13]['source']))

print("=== CELL 14 ===")
print("".join(nb['cells'][14]['source']))

print("=== CELL 15 ===")
print("".join(nb['cells'][15]['source']))
