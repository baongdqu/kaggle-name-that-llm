import json

with open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

infer_src = "".join(nb['cells'][6]['source'])
idx = infer_src.find("spec.get('type') == 'enc'")
if idx != -1:
    print(infer_src[idx:idx+1500])
else:
    print("Not found directly, searching 'enc' in infer:")
    for line in infer_src.split('\n'):
        if 'enc' in line and 'type' in line:
            print(line)

