import json

with open('champion_v65/champion-v65.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

lines = ''.join(nb['cells'][6]['source']).splitlines()
for i, l in enumerate(lines):
    if 'atlas' in l.lower() and ('def ' in l or 'if a.atlas' in l or 'atlas_w' in l or 'atlas_base' in l):
        print(f'{i}: {l}')

