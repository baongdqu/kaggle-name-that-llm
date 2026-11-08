import json, ast

nb = json.load(open('model_arena/arena-benchmark.ipynb', encoding='utf-8'))
print(f"Total cells in Arena Benchmark: {len(nb['cells'])}")

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

assert errors == 0, f"Found {errors} syntax errors in arena-benchmark!"
print("=== ALL ARENA BENCHMARK CODE CELLS PASSED SYNTAX CHECK WITH ZERO ERRORS ===")
