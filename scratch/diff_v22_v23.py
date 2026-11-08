import json

with open("champion_v22/champion-v22.ipynb", "r", encoding="utf-8") as f:
    nb22 = json.load(f)

with open("champion_v23/champion-v23.ipynb", "r", encoding="utf-8") as f:
    nb23 = json.load(f)

print(f"Total cells: V22={len(nb22['cells'])}, V23={len(nb23['cells'])}")

for i in range(len(nb22['cells'])):
    c22 = "".join(nb22['cells'][i].get('source', ''))
    c23 = "".join(nb23['cells'][i].get('source', ''))
    if c22 != c23:
        print(f"\n=================== CELL {i} ({nb22['cells'][i]['cell_type']}) DIFFERENCE ===================")
        import difflib
        diff = difflib.unified_diff(
            c22.splitlines(keepends=True),
            c23.splitlines(keepends=True),
            fromfile=f"v22_cell_{i}",
            tofile=f"v23_cell_{i}",
            n=2
        )
        print("".join(diff)[:1500])
