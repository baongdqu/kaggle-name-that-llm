import json

with open('scratch/public_0639/0-639-teach-more-keep-the-signature.ipynb', 'r', encoding='utf-8') as f:
    nb_kt = json.load(f)

for c_idx in [21, 22, 23, 24]:
    print(f"=== Cell {c_idx} ({nb_kt['cells'][c_idx]['cell_type']}) ===")
    src = ''.join(nb_kt['cells'][c_idx]['source'])
    print(src[:2500])
    print("\n" + "="*50 + "\n")
