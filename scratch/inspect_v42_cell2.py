import json

with open('champion_v42/champion-v42.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(''.join(nb['cells'][2]['source']))

