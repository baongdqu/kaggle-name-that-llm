import json

with open('downloaded_kernels/omerfarukmerey_0663/0-663-0-660-v31-atlas-20.ipynb', 'r', encoding='utf-8') as f:
    nb663 = json.load(f)

comp663 = ''.join(nb663['cells'][8]['source'])
lines = comp663.split('\n')
print(f"Total lines in 663 comp_stack.py: {len(lines)}")
for i, line in enumerate(lines[:50]):
    print(f"{i}: {line}")

