import json

with open('champion_v47/champion-v47.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

comp_src = ''.join(nb['cells'][8]['source'])
idx = comp_src.find('def find_checkpoint')
print(comp_src[idx:idx+800])

