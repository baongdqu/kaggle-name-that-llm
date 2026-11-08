import json
v68 = json.load(open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8'))
infer = ''.join(v68['cells'][6]['source'])
print(infer[46057:])
