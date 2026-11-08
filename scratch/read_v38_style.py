import json
nb = json.load(open('scratch/kchuang_0668/ntl-0668-stack-pool-prompt-rows.ipynb', 'r', encoding='utf-8'))
infer_src = ''.join(nb['cells'][14]['source'])
idx = infer_src.find('failed; using exact inherited prediction:')
print(infer_src[idx-1200:idx+100])
