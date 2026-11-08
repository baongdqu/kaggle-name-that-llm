import json

with open('scratch/gopain_grouping/when-does-test-time-prompt-grouping-help.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [0, 14]:
    print(f"\n{'='*20} CELL {idx} {'='*20}")
    raw = nb['cells'][idx].get('source', '')
    src = "".join(raw) if isinstance(raw, list) else raw
    print(src)
