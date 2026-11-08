import json

with open('champion_v62/champion-v62.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][6]['source'])
lines = src.split('\n')
for i in range(500, 560):
    print(f"{i}: {lines[i]}")

