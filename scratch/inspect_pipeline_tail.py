import json

with open('champion_v30/champion-v30.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = "".join(nb['cells'][9]['source'])
lines = src.split('\n')
for i, l in enumerate(lines[240:320]):
    print(f"{i+241}: {l}")
