import json
nb = json.load(open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8'))
cell12 = ''.join(nb['cells'][12]['source'])
print(cell12)
