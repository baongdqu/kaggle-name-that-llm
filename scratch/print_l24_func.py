import json
nb = json.load(open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8'))
infer_py = ''.join(nb['cells'][6]['source'])
idx = infer_py.find('def l24_predict(')
print(infer_py[idx:idx+3500])
