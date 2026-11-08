import json
v68 = json.load(open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8'))
infer = ''.join(v68['cells'][6]['source'])
idx = infer.find('pred = ')
while idx != -1:
    print('Found pred = at:', idx)
    print(infer[idx:idx+250])
    idx = infer.find('pred = ', idx + 7)
