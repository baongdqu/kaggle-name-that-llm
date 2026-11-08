import json

with open('scratch/kirilltym_639/0-639-teach-more-keep-the-signature.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = "".join(c.get('source', ''))
    if 'night24' in src:
        print(f"Found 'night24' in Cell {i} ({c['cell_type']}):")
        for line in src.splitlines():
            if 'night24' in line:
                print("   ", line[:120])
