import json
import re

with open('scratch/public_top/lightning/0-594-two-stage-lora-olmo2-ensemble.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

src = nb['cells'][1]['source']
if isinstance(src, list):
    src = "".join(src)

# Find kaggle input paths
matches = set(re.findall(r'(/kaggle/input/[^\s\'",\)]+)', src))
print("Matched /kaggle/input paths:")
for m in sorted(matches):
    print(" ", m)

# Find dataset references
ds = set(re.findall(r'([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)', src))
print("\nPossible dataset or model slugs:")
for d in sorted(ds):
    if any(k in d.lower() for k in ['olmo', 'lora', 'two-stage', 'checkpoints', 'base', 'weights', 'allenai']):
        print(" ", d)
