import json

with open("champion_v21/champion-v21.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src = "".join(c.get("source", []))
    first_line = src.splitlines()[0] if src.splitlines() else "(empty)"
    print(f"Cell {i:02d} [{c.get('cell_type')}]: {first_line[:80]}")
