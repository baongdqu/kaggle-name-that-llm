import json

with open('champion_v62/champion-v62.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][6]['source'])
lines = src.split('\n')
for i, l in enumerate(lines):
    if any(k in l for k in ["type == 'enc'", 'type == "enc"', 'enc_predict', 'load_enc', 'modernbert', 'deberta']):
        print(f"{i}: {l}")

