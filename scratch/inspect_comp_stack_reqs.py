import json

with open('champion_v47/champion-v47.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

comp_src = ''.join(nb['cells'][8]['source'])

# Look for paths and requirements in comp_stack.py
lines = comp_src.split('\n')
for line in lines:
    if 'assert' in line or 'glob' in line or 'read_' in line or 'load' in line or 'open(' in line:
        if any(w in line for w in ['DATA', 'POOL', 'test', 'train', 'meta', 'parquet', 'csv', 'heads', 'safetensors']):
            print(line[:120])

