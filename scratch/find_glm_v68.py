import json, re

with open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = "".join(c.get('source', []))
    for m in re.finditer(r'.{0,50}glm.{0,50}', src, re.I):
        print(f"Cell {i}: {m.group(0)}")

