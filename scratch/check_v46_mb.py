import json

with open('champion_v46/champion-v46.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    if 'MODERNBERT' in src or 'modernbert' in src.lower() or 'mb_v5' in src:
        print(f"Cell {i} has mb:")
        for line in src.split('\n'):
            if any(k in line.lower() for k in ['modernbert', 'mb_v5', 'classifier', 'safetensors']):
                print("  ", line[:120])

