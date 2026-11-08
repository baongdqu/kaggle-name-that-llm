import json

with open('train_sovereign_modernbert/train-sovereign-modernbert.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for idx, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    lines = src.split('\n')
    header = lines[0] if lines else ''
    print(f"Cell {idx} ({c.get('cell_type')}): {header[:70]} | length: {len(src)}")

