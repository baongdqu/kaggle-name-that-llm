import json

with open('scratch/lucas_tokattn/name-that-llm-train-tokattn2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = "".join(nb['cells'][0]['source'])
idx = src.find("ids = tok(")
print(src[idx-50:idx+1500])

