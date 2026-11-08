import json

with open('champion_v63/champion-v63.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    src = ''.join(cell['source'])
    if "spec.get('type') == 'enc'" in src or "type" in src and "enc" in src and "def score_enc" in src:
        lines = src.split('\n')
        for i, l in enumerate(lines):
            if "def score_enc" in l or "elif spec['type'] == 'enc'" in l or "spec.get('type') == 'enc'" in l:
                print('\n'.join(lines[max(0, i-2):min(len(lines), i+30)]))
                print('='*50)
