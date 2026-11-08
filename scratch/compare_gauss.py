import json

with open('downloaded_kernels/omerfarukmerey_0663/0-663-0-660-v31-atlas-20.ipynb', 'r', encoding='utf-8') as f:
    nb663 = json.load(f)

with open('champion_v44/champion-v44.ipynb', 'r', encoding='utf-8') as f:
    nb44 = json.load(f)

g663 = ''.join(nb663['cells'][4]['source'])
g44 = ''.join(nb44['cells'][4]['source'])

print(f"gauss 663 len: {len(g663)}, gauss 44 len: {len(g44)}")
print("Equal?", g663 == g44)

