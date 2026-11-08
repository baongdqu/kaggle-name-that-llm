import json

with open('downloaded_kernels/omerfarukmerey_0663/0-663-0-660-v31-atlas-20.ipynb', 'r', encoding='utf-8') as f:
    nb663 = json.load(f)

src663 = ''.join(nb663['cells'][6]['source'])
lines = src663.split('\n')
print(f"Total lines in infer.py: {len(lines)}")
for i, line in enumerate(lines[580:685], start=580):
    print(f"{i}: {line}")

