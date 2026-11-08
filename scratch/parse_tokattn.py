import json, re

with open('scratch/lucas_tokattn/name-that-llm-train-tokattn2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = "".join(nb['cells'][0]['source'])
for line in src.split('\n'):
    if any(k in line for k in ['for ', 'TAG', 'bacc', 'DeBERTa', 'Qwen', 'MODEL', 'MODELS', 'path']):
        print(line)

