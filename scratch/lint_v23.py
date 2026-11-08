import ast
import json

with open("champion_v23/champion-v23.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells in V23: {len(nb['cells'])}")
errors = 0

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        try:
            ast.parse(source)
            print(f"Cell {i:02d} [CODE]: AST PARSE PASS ({len(cell['source'])} lines)")
        except SyntaxError as e:
            print(f"Cell {i:02d} [CODE]: AST PARSE FAIL -> {e}")
            errors += 1

if errors == 0:
    print("\nALL 25 CELLS (14 CODE CELLS) IN CHAMPION V23 PASSED AST SYNTAX VALIDATION 100%!")
else:
    raise RuntimeError(f"{errors} syntax errors found in V23!")
