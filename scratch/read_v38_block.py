import json
nb = json.load(open('scratch/kchuang_0668/ntl-0668-stack-pool-prompt-rows.ipynb', 'r', encoding='utf-8'))
infer_src = ''.join(nb['cells'][14]['source'])
idx = infer_src.find('v34_pairwise_probs.npy')
print(infer_src[idx-400:idx+600])
