import json

with open('champion_v81/champion-v81.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell12 = "".join(nb['cells'][12]['source'])
for line in cell12.split('\n'):
    if 'sibling' in line.lower() or 'adapters' in line.lower():
        print(line)

