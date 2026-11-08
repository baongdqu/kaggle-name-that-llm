import json

with open("champion_v20/champion-v20.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells in V20: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    source = cell['source']
    if isinstance(source, list):
        first_line = source[0].strip() if source else ""
    else:
        first_line = source.splitlines()[0] if source else ""
    print(f"Cell {i:2d} [{ctype:8s}]: {first_line[:80]}")
