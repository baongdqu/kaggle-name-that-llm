import json

nb = json.load(open('scratch/kchuang_0668/ntl-0668-stack-pool-prompt-rows.ipynb', 'r', encoding='utf-8'))
print('Total cells in kchuang:', len(nb['cells']))
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    first_line = c['source'][0] if c['source'] else ''
    if any(k in src.lower() for k in ['0.668', 'verifier', 'pool', 'matched', 'prompt']):
        print(f"Cell {i} ({c['cell_type']}): {first_line[:80].strip()}")

