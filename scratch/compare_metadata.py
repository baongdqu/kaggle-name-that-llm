import json

for v in ['v76', 'v81']:
    with open(f'champion_{v}/kernel-metadata.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    print(f"{v} dataset_sources:", meta.get('dataset_sources', []))
