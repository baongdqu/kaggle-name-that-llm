import json, re

with open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8') as f:
    text = f.read()

matches = re.findall(r'/kaggle/input/[a-zA-Z0-9_\-\./]+', text)
for m in sorted(set(matches)):
    print(m)

