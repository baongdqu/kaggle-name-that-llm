import json

with open('downloaded_kernels/omerfarukmerey_0663/0-663-0-660-v31-atlas-20.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

infer_src = ''.join(nb['cells'][6]['source'])
idx = infer_src.find("spec.get('gauss')")
while idx != -1:
    print(infer_src[idx-50:idx+400])
    print("="*50)
    idx = infer_src.find("spec.get('gauss')", idx+1)

