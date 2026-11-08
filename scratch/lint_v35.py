import json, ast

nb = json.load(open('champion_v35/champion-v35.ipynb', encoding='utf-8'))
print(f"Total cells in V35: {len(nb['cells'])}")

errors = 0
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        lines = [l for l in source.split('\n') if not l.startswith('%')]
        code_to_check = "\n".join(lines)
        try:
            ast.parse(code_to_check)
            print(f"  Cell {idx:02d} [code]: PASS (AST OK)")
        except SyntaxError as e:
            print(f"  Cell {idx:02d} [code]: SYNTAX ERROR: {e}")
            errors += 1

assert errors == 0, f"Found {errors} syntax errors in champion_v35!"
print("=== ALL V35 CODE CELLS PASSED SYNTAX CHECK WITH ZERO ERRORS ===")
