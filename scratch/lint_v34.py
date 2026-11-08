import json, ast

nb = json.load(open('champion_v34/champion-v34.ipynb', encoding='utf-8'))
print(f"Total cells in V34: {len(nb['cells'])}")

errors = 0
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        # strip cell magics for AST check
        lines = [l for l in source.split('\n') if not l.startswith('%')]
        code_to_check = "\n".join(lines)
        try:
            ast.parse(code_to_check)
            print(f"  Cell {idx:02d} [code]: PASS (AST OK)")
        except SyntaxError as e:
            print(f"  Cell {idx:02d} [code]: SYNTAX ERROR: {e}")
            errors += 1

assert errors == 0, f"Found {errors} syntax errors in champion_v34!"
print("=== ALL V34 CODE CELLS PASSED SYNTAX CHECK WITH ZERO ERRORS ===")
