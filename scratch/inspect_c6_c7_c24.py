import json

with open("champion_v27/champion-v27.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== CELL 06 ===")
print("".join(nb["cells"][6]["source"])[:1000])

print("\n=== CELL 07 ===")
print("".join(nb["cells"][7]["source"])[:1500])

print("\n=== CELL 24 ===")
print("".join(nb["cells"][24]["source"])[:2500])
