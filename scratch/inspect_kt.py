import json

with open('scratch/fleongg_0636/ntl-part-two-complement.ipynb', 'r', encoding='utf-8') as f:
    nb_fl = json.load(f)

with open('scratch/public_0639/0-639-teach-more-keep-the-signature.ipynb', 'r', encoding='utf-8') as f:
    nb_kt = json.load(f)

print('fleongg cells:', len(nb_fl['cells']), 'kirilltym cells:', len(nb_kt['cells']))
for i, c in enumerate(nb_kt['cells']):
    src = ''.join(c['source'])
    header = src.splitlines()[0] if src.splitlines() else 'EMPTY'
    print(f"KT Cell {i:02d} ({c['cell_type']}): {header[:90]}")
