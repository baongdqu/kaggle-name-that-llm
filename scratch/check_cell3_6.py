import json

with open('scratch/kirilltym_639/0-639-teach-more-keep-the-signature.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [3, 6]:
    c = nb['cells'][idx]
    print(f"\n{'='*20} CELL {idx} {'='*20}")
    src = "".join(c.get('source', ''))
    print(src[:3000])
