import json
nb = json.load(open('scratch/kchuang_0668/ntl-0668-stack-pool-prompt-rows.ipynb', 'r', encoding='utf-8'))
infer_src = ''.join(nb['cells'][14]['source'])
idx = infer_src.find('_safe_logits = _safe_logits + 0.10 * _gate[:, None] * _pair_log')
print(infer_src[idx-200:idx+400])
