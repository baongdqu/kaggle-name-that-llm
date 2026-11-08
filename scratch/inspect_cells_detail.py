import json

with open("champion_v24/champion-v24.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = "".join(c.get("source", []))
    lines = [l.strip() for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]
    header = lines[0] if lines else "ALL COMMENTS / EMPTY"
    print(f"--- Cell {i:02d} [{c.get('cell_type')}] (lines: {len(src.splitlines())}) ---")
    print("   " + header[:100])
    for line in lines[1:5]:
        print("   " + line[:100])
