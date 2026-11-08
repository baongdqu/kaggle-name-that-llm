import json

with open('downloaded_kernels/omerfarukmerey_0663/0-663-0-660-v31-atlas-20.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell2 = ''.join(nb['cells'][2]['source'])
print(cell2)

