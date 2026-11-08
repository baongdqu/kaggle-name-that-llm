import json

with open('scratch/picklerick_nb/ntl-0-627-encoder-ratio-6-to-1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(nb['cells'][3]['source'])
