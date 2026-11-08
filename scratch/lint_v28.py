import json, ast

with open("champion_v28/champion-v28.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells in V28: {len(nb['cells'])}")
errors = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = "".join(c['source'])
        try:
            ast.parse(src)
        except SyntaxError as e:
            print(f"Cell {i} SyntaxError: {e}")
            errors += 1

if errors == 0:
    print("ALL CODE CELLS IN V28 PARSED CLEANLY WITH AST!")
else:
    print(f"TOTAL ERRORS: {errors}")
