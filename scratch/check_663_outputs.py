import json

with open('downloaded_kernels/omerfarukmerey_0663/0-663-0-660-v31-atlas-20.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Cell 10 output:")
print(json.dumps(nb['cells'][10].get('outputs', []), indent=2))

print("Cell 12 output:")
print(json.dumps(nb['cells'][12].get('outputs', []), indent=2)[:1000])

