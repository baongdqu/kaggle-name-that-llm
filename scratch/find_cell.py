import json

with open('champion_v91/champion-v91.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if '%%writefile infer.py' in src:
        print(f"infer.py is in cell index {idx}, total lines: {len(cell['source'])}")

