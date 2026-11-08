with open('champion_v61/champion-v61.ipynb', 'r', encoding='utf-8') as f:
    import json
    nb = json.load(f)

src = ''.join(nb['cells'][2]['source'])
for line in src.split('\n'):
    if 'glm' in line.lower() or 'gl' in line.lower():
        print(line)

