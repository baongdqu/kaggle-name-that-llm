import json

with open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c6 = ''.join(nb['cells'][6]['source'])
for i, line in enumerate(c6.split('\n')):
    if 'gauss' in line.lower():
        print(f"L{i}: {line[:100]}")

