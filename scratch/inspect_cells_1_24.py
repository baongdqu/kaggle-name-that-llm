import json

with open('scratch/public_0639/0-639-teach-more-keep-the-signature.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [1, 24]:
    print(f"=== Cell {idx} ===")
    print(''.join(nb['cells'][idx]['source']))
    print("="*60)
