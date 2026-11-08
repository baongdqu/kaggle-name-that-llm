import json, os
nb = json.load(open('scratch/kchuang_0668/ntl-0668-stack-pool-prompt-rows.ipynb', 'r', encoding='utf-8'))
print('Checking datasets needed by kchuang:')
# Check if there is kernel-metadata.json
print(os.path.exists('scratch/kchuang_0668/kernel-metadata.json'))
