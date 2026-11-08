import json

with open('downloaded_kernels/alexchilton_0662/name-that-llm-0-662-all-public-what-failed.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(''.join(nb['cells'][1]['source'])[:3000])

