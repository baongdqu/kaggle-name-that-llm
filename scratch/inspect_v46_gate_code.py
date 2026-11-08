import json

with open('champion_v46/champion-v46.ipynb', 'r', encoding='utf-8') as f:
    nb46 = json.load(f)

src46 = ''.join(nb46['cells'][9]['source'])
idx_gate = src46.find('# Sibling Micro-Gates')
print(src46[idx_gate:idx_gate+4000])

