import json
v68 = json.load(open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8'))
infer = ''.join(v68['cells'][6]['source'])
idx = infer.find('linear_sum_assignment(-')
print(infer[idx-300:idx+600])
