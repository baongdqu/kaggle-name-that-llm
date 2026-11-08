import json

with open('scratch/kirilltym_639/0-639-teach-more-keep-the-signature.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells_to_inspect = [1, 3, 6, 7, 11, 20, 21, 22, 24]
for idx in cells_to_inspect:
    c = nb['cells'][idx]
    print(f"\n{'='*20} CELL {idx} ({c['cell_type']}) {'='*20}")
    src = "".join(c.get('source', ''))
    print(src[:2000])
