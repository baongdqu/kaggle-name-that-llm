import ast
import json

with open("champion_v22/champion-v22.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells in V22: {len(nb['cells'])}")
errors = 0
for idx, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        # Filter IPython magic lines starting with % or !
        cleaned_lines = []
        for line in src.splitlines():
            if line.strip().startswith("%") or line.strip().startswith("!"):
                cleaned_lines.append("# " + line)
            else:
                cleaned_lines.append(line)
        cleaned_src = "\n".join(cleaned_lines)
        try:
            ast.parse(cleaned_src)
            print(f"Cell {idx:02d} [code]: PASS ({len(src)} chars)")
        except SyntaxError as e:
            print(f"Cell {idx:02d} [code]: SYNTAX ERROR -> {e}")
            errors += 1

if errors == 0:
    print("ALL CELLS IN V22 PASSED AST LINTER 100%!")
else:
    print(f"FAILED: {errors} syntax errors found!")
