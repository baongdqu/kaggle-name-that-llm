import json

with open("champion_v29/champion-v29.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = "".join(c.get("source", []))
    lines = [l.strip() for l in src.splitlines() if l.strip()]
    first_line = lines[0] if lines else "(empty)"
    print(f"Cell {i:02d} [{c.get('cell_type')}]: {first_line[:90]}")
