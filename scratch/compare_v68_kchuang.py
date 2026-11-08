import json
v68 = json.load(open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8'))
kc = json.load(open('scratch/kchuang_0668/ntl-0668-stack-pool-prompt-rows.ipynb', 'r', encoding='utf-8'))

v68_meta = json.load(open('champion_v68/kernel-metadata.json', 'r', encoding='utf-8'))
print('v68 dataset sources count:', len(v68_meta['dataset_sources']))
