import json

with open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell2 = "".join(nb['cells'][2]['source'])
cell12 = "".join(nb['cells'][12]['source'])

print("=== CELL 2 ===")
print(cell2)
print("=== CELL 12 ===")
print(cell12)

