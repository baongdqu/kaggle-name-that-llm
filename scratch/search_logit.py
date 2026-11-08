import json

with open("champion_v27/champion-v27.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = "".join(c.get("source", []))
    for line_idx, line in enumerate(src.splitlines()):
        if "LOGIT" in line:
            print(f"Cell {i:02d} L{line_idx:03d}: {line.strip()[:100]}")
