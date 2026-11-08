import json

with open('champion_v46/champion-v46.ipynb', 'r', encoding='utf-8') as f:
    nb46 = json.load(f)

src46 = ''.join(nb46['cells'][9]['source'])
idx_hung = src46.find('Adaptive Class-Specific Anchor')
if idx_hung == -1:
    idx_hung = src46.find('linear_sum_assignment')
print(src46[idx_hung-100:idx_hung+2500])

