import json

with open('champion_v63/champion-v63.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c6 = ''.join(nb['cells'][6]['source'])
assert '_dense_clusters = np.unique(Gte[(_cluster_sizes >= 5) & (_cluster_sizes <= 18)])' in c6, 'Threshold >= 5 not found in infer.py'
assert 'linear_sum_assignment' in c6, 'linear_sum_assignment not found'

c12 = ''.join(nb['cells'][12]['source'])
assert "--sinkhorn', '0.45" in c12, 'Sinkhorn not 0.45'
assert '"w": 1.0' in c12, 'Qwen2.5-14B not w=1.0'

print('Champion V63 notebook assertions PASSED flawlessly!')

