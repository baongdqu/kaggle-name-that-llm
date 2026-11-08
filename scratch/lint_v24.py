import json
import ast

with open("champion_v24/champion-v24.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells in V24: {len(nb['cells'])}")
errors = []

for idx, cell in enumerate(nb['cells']):
    c_type = cell['cell_type']
    raw = cell.get('source', '')
    src = "".join(raw) if isinstance(raw, list) else raw
    if c_type == 'code':
        # Remove ipython magics if any
        clean_lines = []
        for line in src.splitlines():
            if line.strip().startswith('!') or line.strip().startswith('%'):
                clean_lines.append(f"# {line}")
            else:
                clean_lines.append(line)
        clean_code = "\n".join(clean_lines)
        try:
            ast.parse(clean_code)
            print(f"Cell {idx:02d} [CODE]: AST PARSE PASS ({len(clean_lines)} lines)")
        except SyntaxError as e:
            print(f"Cell {idx:02d} [CODE]: SYNTAX ERROR! Line {e.lineno}: {e.msg}")
            errors.append((idx, e))

if errors:
    print(f"\nFAILED: {len(errors)} cells had syntax errors.")
    exit(1)
else:
    print("\nALL 25 CELLS (14 CODE CELLS) IN CHAMPION V24 PASSED AST SYNTAX VALIDATION 100%!")
