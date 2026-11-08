import json
import ast

with open("champion_v21/champion-v21.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

errors = []
print(f"Linting {len(nb['cells'])} cells in champion_v21/champion-v21.ipynb ...")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = cell['source']
        if isinstance(source, list):
            code_str = "".join(source)
        else:
            code_str = source
        try:
            ast.parse(code_str)
            print(f"  Cell {i:2d}: Syntax AST OK ({len(code_str.splitlines())} lines)")
        except SyntaxError as e:
            print(f"  Cell {i:2d}: SYNTAX ERROR! {e}")
            errors.append((i, e))

if not errors:
    print("=== [100% PASS] ALL CODE CELLS PASSED AST SYNTAX CHECK! ===")
else:
    print(f"=== FAILED: {len(errors)} cells had syntax errors! ===")
