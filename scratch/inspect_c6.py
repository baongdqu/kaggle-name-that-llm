import json

with open('champion_v63/champion-v63.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c6 = ''.join(nb['cells'][6]['source'])
lines = c6.split('\n')
print(f"Total lines in Cell 6: {len(lines)}")

for i, l in enumerate(lines):
    if any(k in l for k in ['_LEN_BIAS', 'wlen', 'word_len', 'hungarian', 'linear_sum_assignment', 'adapt_thr', 'min_cluster_size', 'center_groups']):
        print(f"L{i}: {l[:100]}")

