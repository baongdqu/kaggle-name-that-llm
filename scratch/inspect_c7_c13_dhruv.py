import json

with open("scratch/dhruv_repro/name-that-llm-fully-reproducible.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in [7, 13]:
    c = nb["cells"][idx]
    print(f"\n==================== CELL {idx} ====================")
    print("".join(c["source"]))
