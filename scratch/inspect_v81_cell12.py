import json

with open('champion_v81/champion-v81.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src12 = "".join(nb['cells'][12]['source'])
print("Cell 12 source:")
print(src12)

