import json

with open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = "".join(nb['cells'][6]['source'])
idx = src.find("spec.get('type') == 'enc'")
print(src[idx+800:idx+2200])

