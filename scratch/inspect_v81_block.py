import json

with open('champion_v81/champion-v81.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = nb['cells'][6]['source'][0]
lines = src.split('\n')
for i, line in enumerate(lines):
    if 'V81 ARCHITECTURE' in line:
        print("Found V81 block around line", i)
        for j in range(max(0, i-5), min(len(lines), i+60)):
            print(f"{j:4d}: {lines[j]}")
        break

