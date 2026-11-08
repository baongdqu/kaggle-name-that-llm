import ast
import json
import sys

with open("champion_v27/champion-v27.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells in V27: {len(nb['cells'])}")
errors = 0

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        try:
            ast.parse(src)
            print(f"Cell {i:2d} (code, {len(src):5d} chars): PASS")
        except SyntaxError as e:
            print(f"Cell {i:2d} SYNTAX ERROR: {e}")
            errors += 1

if errors == 0:
    print("\nALL CODE CELLS IN CHAMPION V27 PASSED AST PARSE 100%!")
else:
    print(f"\nFAILED with {errors} errors!")
    sys.exit(1)
