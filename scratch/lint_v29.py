import json, ast

with open("champion_v29/champion-v29.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells in V29: {len(nb['cells'])}")
errors = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = "".join(c['source'])
        # strip ipython magics for AST check
        clean_lines = [l for l in src.splitlines() if not l.strip().startswith("!") and not l.strip().startswith("%")]
        clean_src = "\n".join(clean_lines)
        try:
            ast.parse(clean_src)
        except SyntaxError as e:
            print(f"Cell {i} SyntaxError: {e}")
            errors += 1

if errors == 0:
    print("ALL CODE CELLS IN V29 PARSED CLEANLY WITH AST!")
else:
    print(f"TOTAL ERRORS: {errors}")
