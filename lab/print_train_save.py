import json

with open('train_sovereign_modernbert/train-sovereign-modernbert.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][5]['source'])
lines = src.split('\n')
for i, l in enumerate(lines):
    if 'save' in l.lower() or 'checkpoint' in l.lower() or 'out' in l.lower():
        print(f"{i}: {l}")

