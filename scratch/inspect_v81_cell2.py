import json

with open('champion_v81/champion-v81.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell2 = "".join(nb['cells'][2]['source'])
for line in cell2.split('\n'):
    if any(k in line for k in ['SIBLING_LORA', 'find_sibling_lora', 'heads.pt', 'meta.json']):
        print(line)

