import json
import ast
from pathlib import Path

with open('scratch/public_top/lightning/0-594-two-stage-lora-olmo2-ensemble.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

src = nb['cells'][1]['source']
if isinstance(src, list):
    src = "".join(src)

tree = ast.parse(src)
out_dir = Path('scratch/olmo_extracted')
out_dir.mkdir(exist_ok=True)

for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'FILES':
                if isinstance(node.value, ast.Dict):
                    for k, v in zip(node.value.keys, node.value.values):
                        fname = k.value
                        fcontent = v.value
                        (out_dir / fname).write_text(fcontent, encoding='utf-8')
                        print(f"Extracted {fname} ({len(fcontent)} chars)")
