import json

with open('downloaded_kernels/omerfarukmerey_0663/0-663-0-660-v31-atlas-20.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== CELL 2 ===")
print(''.join(nb['cells'][2]['source'])[:1500])

print("\n=== CELL 10 ===")
print(''.join(nb['cells'][10]['source']))

print("\n=== CELL 12 ===")
print(''.join(nb['cells'][12]['source']))
