import json

with open('champion_v65/champion-v65.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

lines = ''.join(nb['cells'][6]['source']).splitlines()
for i, l in enumerate(lines):
    if 'models' in l.lower() and ('for ' in l or 'load' in l):
        print(f'{i}: {l}')

