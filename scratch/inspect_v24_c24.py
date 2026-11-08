import json

with open('champion_v24/champion-v24.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = "".join(nb['cells'][24]['source'])
print(src[:2500])
