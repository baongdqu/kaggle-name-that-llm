import json, shutil, os

os.makedirs('champion_v74', exist_ok=True)

# 1. Prepare kernel-metadata.json
with open('champion_v68/kernel-metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

meta['id'] = "baonguyenduyquoc/ntl-champion-v74-tokattn-apex"
meta['title'] = "NTL Champion V74 TokAttn Apex"
meta['code_file'] = "champion-v74.ipynb"

# Ensure swaroopmeher/ntl-public-heads is in dataset_sources (or already there)
if "swaroopmeher/ntl-public-heads" not in meta['dataset_sources']:
    meta['dataset_sources'].append("swaroopmeher/ntl-public-heads")

with open('champion_v74/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)

print("Created champion_v74/kernel-metadata.json successfully!")
print("Dataset sources count:", len(meta['dataset_sources']))
print("Kernel sources count:", len(meta['kernel_sources']))

