import json

with open('train_sovereign_modernbert/train-sovereign-modernbert.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(''.join(nb['cells'][2]['source']))

