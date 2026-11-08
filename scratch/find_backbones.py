import json

with open('scratch/lucas_tokattn/name-that-llm-train-tokattn2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = "".join(nb['cells'][0]['source'])
idx = src.find("BACKBONES =")
if idx != -1:
    print(src[idx:idx+500])
else:
    print("BACKBONES not found directly, searching...")
    for line in src.split('\n'):
        if 'BACKBONES' in line:
            print(line)

