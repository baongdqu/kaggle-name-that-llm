import json

with open("champion_v27/champion-v27.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== CELL 21 ===")
print("".join(nb["cells"][21]["source"]))

print("\n=== CELL 22 ===")
print("".join(nb["cells"][22]["source"]))
