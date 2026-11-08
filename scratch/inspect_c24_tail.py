import json

with open("champion_v27/champion-v27.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

src24 = "".join(nb["cells"][24]["source"])
print(f"Total lines in Cell 24: {len(src24.splitlines())}")
# Print lines from line 40 onwards
lines = src24.splitlines()
for idx, l in enumerate(lines[35:], start=36):
    print(f"{idx:03d}: {l}")
