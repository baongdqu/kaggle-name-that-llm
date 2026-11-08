import json, re

with open('downloaded_kernels/omerfarukmerey_0663/0-663-0-660-v31-atlas-20.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

infer_src = ''.join(nb['cells'][6]['source'])
idx = infer_src.find('dense-cluster Hungarian')
print(infer_src[max(0, idx-800):min(len(infer_src), idx+800)])

