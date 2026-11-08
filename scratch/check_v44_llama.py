import json

with open('champion_v44/champion-v44.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'BL' in src or 'llama' in src.lower():
        print(f"Cell {i} has llama/BL:")
        for line in src.split('\n'):
            if 'BL' in line or 'llama' in line.lower() or 'mdir' in line:
                print("  ", line[:120])

