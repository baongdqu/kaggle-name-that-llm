import json, re

with open('downloaded_kernels/omerfarukmerey_0663/0-663-0-660-v31-atlas-20.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

infer_src = ''.join(nb['cells'][6]['source'])
matches = [m.start() for m in re.finditer(r'linear_sum_assignment', infer_src)]
print(f"Occurrences of linear_sum_assignment: {len(matches)}")
for idx in matches:
    print(infer_src[max(0, idx-100):min(len(infer_src), idx+300)])
    print("-" * 50)

