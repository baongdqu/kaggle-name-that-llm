import json

with open('train_sovereign_modernbert/train-sovereign-modernbert.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [3, 4, 5]:
    print(f"=== Cell {idx} ===")
    print(''.join(nb['cells'][idx]['source'])[:500])

