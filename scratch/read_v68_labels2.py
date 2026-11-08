import json
nb = json.load(open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8'))
infer_py = ''.join(nb['cells'][6]['source'])
for i, line in enumerate(infer_py.splitlines()):
    if 'LABELS = ' in line:
        print('\n'.join(infer_py.splitlines()[i:i+6]))
        break
