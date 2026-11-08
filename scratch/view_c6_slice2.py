import json

with open('champion_v63/champion-v63.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c6 = ''.join(nb['cells'][6]['source'])
lines = c6.split('\n')
for i in range(690, min(len(lines), 770)):
    print(f"L{i:03d}: {lines[i]}")

