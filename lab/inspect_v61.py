import json

with open('champion_v61/champion-v61.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb.get('cells', [])
print(f'Total cells: {len(cells)}')
for i, c in enumerate(cells):
    src = ''.join(c.get('source', []))
    lines = src.split('\n')
    header = lines[0] if lines else ''
    low = src.lower()
    if 'glm' in low or 'sinkhorn' in low or 'qwen' in low or 'modernbert' in low or 'hungarian' in low or 'munkres' in low:
        print(f"Cell {i} ({c.get('cell_type')}): {header[:70]} | chars: {len(src)}")

