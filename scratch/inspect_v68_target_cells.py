import json, re

with open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 2: Setup & Discovery
cell2_src = "".join(nb['cells'][2]['source'])
print("Cell 2 lines:", len(cell2_src.split('\n')))

# Cell 6: infer.py
cell6_src = "".join(nb['cells'][6]['source'])
print("Cell 6 lines:", len(cell6_src.split('\n')))

# Cell 12: MODELS & execution
cell12_src = "".join(nb['cells'][12]['source'])
print("Cell 12 lines:", len(cell12_src.split('\n')))

