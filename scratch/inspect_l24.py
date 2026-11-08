import json
nb = json.load(open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8'))
infer_py = ''.join(nb['cells'][6]['source'])
idx = infer_py.find(" spec.get type == l24 \)
print(infer_py[idx:idx+2500])
