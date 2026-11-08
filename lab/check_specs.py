import json

with open('champion_v62/champion-v62.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][6]['source'])
lines = src.split('\n')
for i, l in enumerate(lines):
    if any(k in l for k in ["s.get('type')", "s['type']", 'l24', 'MODELS', 'specs']):
        print(f"{i}: {l}")

