import json

with open('champion_v47/champion-v47.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

infer_src = ''.join(nb['cells'][6]['source'])
idx = infer_src.find("spec.get('type') == 'enc'")
print(infer_src[idx-50:idx+1200])

