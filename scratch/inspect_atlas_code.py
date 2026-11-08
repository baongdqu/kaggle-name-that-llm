import json

with open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

infer_src = "".join(nb['cells'][6]['source'])
idx = infer_src.find("if a.atlas_dir and os.path.exists(a.atlas_dir)")
if idx == -1:
    idx = infer_src.find("atlas_dir")
print(infer_src[idx:idx+2500])
