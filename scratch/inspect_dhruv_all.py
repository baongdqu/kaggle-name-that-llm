import json

with open("scratch/dhruv_repro/name-that-llm-fully-reproducible.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells in Dhruv Repro: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src = "".join(c.get("source", []))
    lines = [l.strip() for l in src.splitlines() if l.strip()]
    header = lines[0] if lines else "(empty)"
    print(f"Cell {i:02d} [{c.get('cell_type')}]: {header[:120]}")
