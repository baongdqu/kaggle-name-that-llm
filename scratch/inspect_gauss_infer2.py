import json

with open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c6 = ''.join(nb['cells'][6]['source'])
lines = c6.split('\n')
for i in range(435, 455):
    print(f"L{i}: {lines[i]}")

