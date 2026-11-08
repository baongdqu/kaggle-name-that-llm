import json

for v in ['v77', 'v78', 'v79', 'v80']:
    path = f'champion_{v}/champion-{v}.ipynb'
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"[{v.upper()}] Cells: {len(nb['cells'])}, valid JSON.")

