import json

with open('downloaded_kernels/omerfarukmerey_0663/0-663-0-660-v31-atlas-20.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    print(f"Cell {i} ({c['cell_type']}, len={len(src)})")
    if i in [0, 1, 2, 3, 5, 7, 9, 10, 11, 12]:
        print(src[:400])
        print("="*60)

