import json

with open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c4 = ''.join(nb['cells'][4]['source'])
print("Total lines in gauss.py:", len(c4.split('\n')))
print(c4[:1500])

