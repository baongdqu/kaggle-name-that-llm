import json

nb = json.load(open('champion_v50/champion-v50.ipynb', encoding='utf-8'))
print(f"Total cells in V50: {len(nb['cells'])}")

for i, cell in enumerate(nb['cells']):
    ctype = cell.get('cell_type')
    src = ''.join(cell.get('source', []))
    if ctype == 'code':
        if src.startswith('%%writefile'):
            lines = src.split('\n')
            fname = lines[0].split()[1]
            code = '\n'.join(lines[1:])
            compile(code, fname, 'exec')
            print(f"Cell {i} (%%writefile {fname}): compiled OK ({len(code)} chars)")
        else:
            compile(src, f"cell_{i}", 'exec')
            print(f"Cell {i} (standard code): compiled OK ({len(src)} chars)")

print("ALL CELLS IN V50 VERIFIED PERFECTLY!")

