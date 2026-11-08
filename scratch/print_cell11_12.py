import json

nb = json.load(open('scratch/kchuang_0668/ntl-0668-stack-pool-prompt-rows.ipynb', 'r', encoding='utf-8'))
for i in [11, 12]:
    print(f"=== CELL {i} ===")
    print(''.join(nb['cells'][i]['source']))

