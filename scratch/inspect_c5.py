import json

with open('train_sovereign_modernbert_31k/train-sovereign-modernbert-31k.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Let's inspect Cell 5
cell5_src = ''.join(nb['cells'][5]['source'])
print('Cell 5 length:', len(cell5_src))
print(cell5_src[:1000])

