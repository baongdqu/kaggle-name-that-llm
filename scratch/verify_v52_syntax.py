import json

with open('champion_v52/champion-v52.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        clean_lines = []
        for line in source.split('\n'):
            if line.strip().startswith('!') or line.strip().startswith('%%') or line.strip().startswith('%'):
                clean_lines.append('# ' + line)
            else:
                clean_lines.append(line)
        clean_code = '\n'.join(clean_lines)
        try:
            compile(clean_code, f'cell_{i}', 'exec')
            print(f"Cell {i:2d} [code]: SYNTAX VALID ({len(source)} chars)")
        except SyntaxError as e:
            print(f"Cell {i:2d} [code]: SYNTAX ERROR -> {e}")
            raise
    else:
        print(f"Cell {i:2d} [markdown]: OK")

print("ALL CELLS IN V52 COMPILED SUCCESSFULLY WITHOUT ANY SYNTAX ERROR!")

