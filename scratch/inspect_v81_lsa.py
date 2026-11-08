import json

with open('champion_v81/champion-v81.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = nb['cells'][6]['source'][0]
lines = src.split('\n')
for j in range(817, 836):
    print(f"{j:4d}: {lines[j]}")

