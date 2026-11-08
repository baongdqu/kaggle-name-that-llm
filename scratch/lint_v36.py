import json, ast, re

nb_path = 'champion_v36/champion-v36.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in Champion V36: {len(nb['cells'])}")
errors = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        code = "".join(cell['source'])
        # strip cell magic like %%writefile
        code_lines = [l for l in code.splitlines() if not l.strip().startswith('%%')]
        clean_code = "\n".join(code_lines)
        try:
            ast.parse(clean_code)
            print(f"  Cell {i:02d} [code]: PASS (AST OK)")
        except SyntaxError as e:
            print(f"  Cell {i:02d} [code]: SYNTAX ERROR -> {e}")
            errors += 1

assert errors == 0, f"Found {errors} syntax errors in Champion V36!"
print("=== ALL CHAMPION V36 CODE CELLS PASSED SYNTAX CHECK WITH ZERO ERRORS ===")
