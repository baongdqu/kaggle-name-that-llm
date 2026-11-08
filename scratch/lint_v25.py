import ast
import json

with open("champion_v25/champion-v25.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells in V25: {len(nb['cells'])}")
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
print(f"Total code cells: {len(code_cells)}")

errors = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        try:
            ast.parse(source)
            print(f"Cell {i} (code): PASS")
        except SyntaxError as e:
            print(f"Cell {i} (code): SYNTAX ERROR -> {e}")
            errors.append((i, e))

if errors:
    print(f"\nFAILED: {len(errors)} syntax errors found!")
else:
    print("\nSUCCESS: All V25 cells passed AST syntax check 100%!")
