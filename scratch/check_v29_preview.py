import json

with open("champion_v29/champion-v29.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = "".join(c.get("source", []))
    for line_idx, line in enumerate(src.splitlines()):
        if any(k in line for k in ["TINY", "PUBLIC_PREVIEW", "len(test)", "preview", "test.csv"]):
            print(f"Cell {i:02d} L{line_idx:03d}: {line.strip()[:100]}")
