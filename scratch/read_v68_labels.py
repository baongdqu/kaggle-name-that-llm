import json
nb = json.load(open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8'))
infer_py = ''.join(nb['cells'][6]['source'])
for line in infer_py.splitlines()[:50]:
    if 'LABELS' in line or 'classes' in line:
        print(line)
