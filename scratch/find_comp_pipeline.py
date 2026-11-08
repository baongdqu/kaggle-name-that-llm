import json

with open('champion_v30/champion-v30.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = "".join(nb['cells'][9]['source'])
idx = 0
while True:
    pos = src.find('comp_', idx)
    if pos == -1:
        break
    print(src[max(0, pos-100):min(len(src), pos+200)])
    print("-" * 50)
    idx = pos + 1
