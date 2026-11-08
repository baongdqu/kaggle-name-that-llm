import json

with open('champion_v62/champion-v62.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c12 = ''.join(nb['cells'][12]['source'])
assert "--sinkhorn', '0.45" in c12, 'Sinkhorn not 0.45'
assert '"name": "q25_14", "type": "l24", "path": B25_14, "adapters": [Q25_14], "keep": Q25_14C[\'keep\'], "layers": Q25_14C[\'layers\'], "w": 1.0' in c12, 'Qwen2.5-14B not w=1.0'

c6 = ''.join(nb['cells'][6]['source'])
assert 'V62 Anchor-Hungarian' in c6, 'Anchor-Hungarian not in infer.py'
assert '_lsa' in c6, 'Linear sum assignment not in infer.py'
assert '_margins >= 0.20' in c6, 'Margin threshold not in infer.py'
assert '_top_p >= 0.35' in c6, 'Probability threshold not in infer.py'

print('All Champion V62 assertions PASSED flawlessly!')

