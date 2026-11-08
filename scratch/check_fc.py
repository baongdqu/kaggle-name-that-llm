import json

with open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c2 = ''.join(nb['cells'][2]['source'])
for line in c2.split('\n')[:30]:
    if 'FAST_COMMIT' in line:
        print(line)

