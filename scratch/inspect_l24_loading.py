import json

with open('champion_v81/champion-v81.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = nb['cells'][6]['source'][0]
lines = src.split('\n')
for i, line in enumerate(lines):
    if 'adapters' in line or 'load_adapters' in line or 'l24' in line:
        if any(w in line for w in ['for', 'def', 'class', 'torch.load', 'head']):
            print(f"{i:4d}: {line}")

