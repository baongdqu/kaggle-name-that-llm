import json

with open("champion_v21/champion-v21.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i in [0, 1, 2, 22, 23, 24]:
    if i < len(nb['cells']):
        c = nb['cells'][i]
        src = "".join(c.get("source", []))
        print(f"--- Cell {i} ({c.get('cell_type')}) ---")
        print(src[:300])
        print("...\n")
