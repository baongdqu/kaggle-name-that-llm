with open('champion_v61/champion-v61.ipynb', 'r', encoding='utf-8') as f:
    import json
    nb = json.load(f)

for idx, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if 'glm' in src.lower():
        print(f"Cell {idx}:")
        for line in src.split('\n'):
            if 'glm' in line.lower():
                print("  ", line)

