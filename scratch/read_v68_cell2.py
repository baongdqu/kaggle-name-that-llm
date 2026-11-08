import json
nb = json.load(open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8'))
cell2 = ''.join(nb['cells'][2]['source'])
print(cell2)
