import json

with open('champion_v24/champion-v24.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in V24: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src = "".join(c['source'])
    if any(k in src.lower() for k in ['sibling', 'micro_gate', 'luna', 'gemini-3.8']):
        print(f"Cell {i} ({c['cell_type']}): len={len(src)}")
        lines = [l for l in c['source'] if any(k in l.lower() for k in ['sibling', 'micro_gate', 'luna', 'gemini-3.8'])]
        for l in lines[:5]:
            print("  ", l.strip())
