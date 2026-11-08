import json

with open('downloaded_kernels/omerfarukmerey_0663/0-663-0-660-v31-atlas-20.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== CELL 4 (gauss.py) ===")
print(''.join(nb['cells'][4]['source'])[:500])

print("\n=== CELL 8 (comp_stack.py) ===")
comp_src = ''.join(nb['cells'][8]['source'])
print(f"comp_stack length: {len(comp_src)}")
print(comp_src[:1000])

print("\n=== CELL 6 (infer.py) ===")
infer_src = ''.join(nb['cells'][6]['source'])
print(f"infer.py length: {len(infer_src)}")
print("infer.py head:")
print(infer_src[:1000])
print("\ninfer.py tail (last 1500 chars):")
print(infer_src[-1500:])

