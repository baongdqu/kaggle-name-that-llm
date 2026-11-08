import json

with open('champion_v30/champion-v30.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== Cell 13 ===")
print("".join(nb['cells'][13]['source']))
