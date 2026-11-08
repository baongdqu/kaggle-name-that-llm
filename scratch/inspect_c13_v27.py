import json

with open("champion_v27/champion-v27.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

c13 = nb["cells"][13]
print("".join(c13["source"]))
