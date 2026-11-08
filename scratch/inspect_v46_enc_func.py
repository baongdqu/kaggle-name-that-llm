import json

with open('champion_v46/champion-v46.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][9]['source'])
idx = src.find('Fully fine-tuned encoder')
print(src[idx-100:idx+1500])

