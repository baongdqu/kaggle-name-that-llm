import json

with open('champion_v30/champion-v30.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = "".join(nb['cells'][9]['source'])
for kw in ['sibling', 'luna', 'micro_gate', 'W_SVC', 'tau', 'sinkhorn']:
    pos = src.lower().find(kw)
    print(f"Keyword '{kw}': {'FOUND at ' + str(pos) if pos != -1 else 'NOT FOUND'}")
