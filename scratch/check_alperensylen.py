import json

with open('downloaded_kernels/alperensylen_atlas20/exp25-v31-llamafree-atlas20-sota.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"alperensylen cell count: {len(nb['cells'])}")
print("=== Cell 0 ===")
print(''.join(nb['cells'][0]['source'])[:1000])

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'MODELS =' in src:
        print(f"=== Cell {i} (MODELS) ===")
        print(src[:1000])

