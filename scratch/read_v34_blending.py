import json
nb = json.load(open('scratch/kchuang_0668/ntl-0668-stack-pool-prompt-rows.ipynb', 'r', encoding='utf-8'))
infer_src = ''.join(nb['cells'][14]['source'])
for line in infer_src.splitlines():
    if any(k in line for k in ['v34', 'pairwise', 'probs', 'pair']):
        print(line)
