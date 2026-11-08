import json
import re

with open('scratch/public_top/mtoshi/test-conditional-author-decoding.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Mtoshi total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells'][:10]):
    src = cell.get('source', '')
    if isinstance(src, list):
        src = "".join(src)
    print(f"Cell {i} ({cell.get('cell_type')}): {len(src)} chars | First 100 chars: {src[:100]!r}")

all_src = ""
for cell in nb['cells']:
    s = cell.get('source', '')
    if isinstance(s, list):
        s = "".join(s)
    all_src += s + "\n"

ds = set(re.findall(r'([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)', all_src))
print("\nPossible dataset or model slugs in mtoshi:")
for d in sorted(ds):
    if any(k in d.lower() for k in ['author', 'decoding', 'conditional', 'lora', 'two-stage', 'weights', 'qwen', 'pool']):
        print(" ", d)
