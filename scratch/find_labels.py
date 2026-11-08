import json

with open('champion_v63/champion-v63.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c6 = ''.join(nb['cells'][6]['source'])
for line in c6.split('\n')[:50]:
    if 'LABELS' in line or 'labels' in line:
        print(line)

