import json

with open('scratch/lucas_enc2/name-that-llm-extract-encoders2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = "".join(nb['cells'][0]['source'])
idx = src.find("ENCODERS =")
print(src[idx:idx+1500])

