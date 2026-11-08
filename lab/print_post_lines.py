with open('champion_v61/champion-v61.ipynb', 'r', encoding='utf-8') as f:
    import json
    nb = json.load(f)

src = ''.join(nb['cells'][6]['source'])
lines = src.split('\n')
for i in range(635, min(len(lines), 675)):
    print(f"{i}: {lines[i]}")

