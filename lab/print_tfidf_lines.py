import json

with open('champion_v62/champion-v62.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][6]['source'])
lines = src.split('\n')
for i in range(200, 320):
    if 'tfidf' in lines[i].lower() or 'extra' in lines[i].lower() or 'svc' in lines[i].lower():
        print(f"{i}: {lines[i]}")

