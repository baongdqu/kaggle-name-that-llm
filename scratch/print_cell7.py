import json

with open('scratch/public_0639/0-639-teach-more-keep-the-signature.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== Cell 7 ===")
print(''.join(nb['cells'][7]['source']))
