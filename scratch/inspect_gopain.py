import json

with open('scratch/gopain_grouping/when-does-test-time-prompt-grouping-help.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    raw = c.get('source', '')
    src = "".join(raw) if isinstance(raw, list) else raw
    first_line = [l.strip() for l in src.splitlines() if l.strip()]
    fl = first_line[0] if first_line else "EMPTY"
    print(f"Cell {i:02d} [{c['cell_type']}]: {fl[:90]}")
