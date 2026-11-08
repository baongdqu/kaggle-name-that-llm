import json

with open('scratch/kirilltym_639/0-639-teach-more-keep-the-signature.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src24 = "".join(nb['cells'][24].get('source', ''))
print(f"Cell 24 length: {len(src24)}")
print(src24[-2500:])
