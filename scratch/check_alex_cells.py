import json

with open('downloaded_kernels/alexchilton_0662/name-that-llm-0-662-all-public-what-failed.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== Alex Cell 5 (Setup) ===")
print(''.join(nb['cells'][5]['source'])[:1500])

print("=== Alex Cell 13 (Comp stack) ===")
print(''.join(nb['cells'][13]['source']))

print("=== Alex Cell 15 (Members & Run) ===")
print(''.join(nb['cells'][15]['source'])[:1500])

