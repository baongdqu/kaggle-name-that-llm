import json

with open('champion_v91/champion-v91.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = nb['cells'][6]['source']
if isinstance(src, list):
    lines = src
else:
    lines = src.splitlines(keepends=True)

for i in range(880, min(900, len(lines))):
    print(f"{i+1}: {repr(lines[i])}")

