import json

with open('scratch/public_top/lightning/0-594-two-stage-lora-olmo2-ensemble.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

src = nb['cells'][1]['source']
if isinstance(src, list):
    src = "".join(src)

# Let's extract the FILES dict keys
import ast
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'FILES':
                if isinstance(node.value, ast.Dict):
                    keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
                    print("FILES keys:", keys)
