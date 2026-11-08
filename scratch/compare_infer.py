import json

with open('downloaded_kernels/omerfarukmerey_0663/0-663-0-660-v31-atlas-20.ipynb', 'r', encoding='utf-8') as f:
    nb663 = json.load(f)

with open('champion_v44/champion-v44.ipynb', 'r', encoding='utf-8') as f:
    nb44 = json.load(f)

src663 = ''.join(nb663['cells'][6]['source'])
src44 = ''.join(nb44['cells'][5]['source'])

print(f"663 infer.py len: {len(src663)}")
print(f"V44 infer.py len: {len(src44)}")

# Check arguments parsed
for line in src663.split('\n'):
    if 'add_argument' in line:
        print("663 arg:", line.strip())

