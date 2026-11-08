import json

with open("scratch/dhruv_repro/name-that-llm-fully-reproducible.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in [0, 10, 11, 12, 13, 15]:
    c = nb["cells"][idx]
    print(f"\n==================== CELL {idx} [{c['cell_type']}] ====================")
    print("".join(c["source"]))
