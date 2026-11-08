import json

with open('downloaded_kernels/omerfarukmerey_0663/0-663-0-660-v31-atlas-20.ipynb', 'r', encoding='utf-8') as f:
    nb663 = json.load(f)

src663 = ''.join(nb663['cells'][6]['source'])
for i, line in enumerate(src663.split('\n')):
    if 'dp' in line.lower() or 'cuda:1' in line.lower() or 'device_map' in line.lower():
        print(f"Line {i}: {line[:120]}")

