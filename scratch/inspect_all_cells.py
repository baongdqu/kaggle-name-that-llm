import json

with open('scratch/public_0639/0-639-teach-more-keep-the-signature.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    lines = src.splitlines()
    print(f"Cell {i:02d} ({c['cell_type']}, {len(lines)} lines):")
    for l in lines[:4]:
        print("  ", l[:100])
    print()
