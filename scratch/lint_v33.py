import json
import ast
import sys

nb_path = 'champion_v33/champion-v33.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in {nb_path}: {len(nb['cells'])}")
errors = 0

for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        source = "".join(cell.get('source', []))
        # Remove shell commands or ipython magics if any for ast parsing
        clean_lines = []
        for line in source.split('\n'):
            stripped = line.strip()
            if stripped.startswith('!') or stripped.startswith('%'):
                clean_lines.append('# ' + line)
            else:
                clean_lines.append(line)
        clean_source = '\n'.join(clean_lines)
        try:
            ast.parse(clean_source)
            print(f"Cell {i:02d}: Syntax OK ({len(source.splitlines())} lines)")
        except SyntaxError as e:
            print(f"Cell {i:02d}: SYNTAX ERROR: {e}")
            errors += 1

if errors == 0:
    print("ALL CODE CELLS PASSED SYNTAX CHECK (0 ERRORS)!")
    sys.exit(0)
else:
    print(f"FOUND {errors} SYNTAX ERRORS!")
    sys.exit(1)
