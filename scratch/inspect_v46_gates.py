import json

with open('champion_v46/champion-v46.ipynb', 'r', encoding='utf-8') as f:
    nb46 = json.load(f)

for i, cell in enumerate(nb46['cells']):
    src = ''.join(cell.get('source', []))
    if 'hepta' in src.lower() or 'gate' in src.lower():
        print(f"Cell {i} has gates:")
        for line in src.split('\n'):
            if any(k in line.lower() for k in ['gate', 'astra', 'luna', 'sol', 'terra', 'fable', 'sonnet', 'flash']):
                print("  ", line[:120])

