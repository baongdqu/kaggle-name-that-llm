import json

with open('scratch/typoguard_nb/0-624-two-stage-lora-encoders-typoguard.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(nb['cells'][17]['source'])
