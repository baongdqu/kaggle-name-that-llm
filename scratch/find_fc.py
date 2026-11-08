import json

with open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    if 'FAST_COMMIT' in src:
        print(f"Cell {i} has FAST_COMMIT:")
        for line in src.split('\n'):
            if 'FAST_COMMIT' in line:
                print(f"  {line}")

