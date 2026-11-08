import json

with open('champion_v63/champion-v63.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c12 = ''.join(nb['cells'][12]['source'])
print(c12)

