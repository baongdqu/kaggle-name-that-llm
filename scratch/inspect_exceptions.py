import json

for v in ['v81', 'v82']:
    with open(f'champion_{v}/champion-{v}.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"=== {v.upper()} ===")
    src = nb['cells'][6]['source'][0]
    print("Contains except Exception:", "except Exception as _e:" in src)
    # Check if there is any code outside try-except
    print("Lines of infer.py:", len(src.split('\n')))

