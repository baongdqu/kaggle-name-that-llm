with open('champion_v61/champion-v61.ipynb', 'r', encoding='utf-8') as f:
    import json
    nb = json.load(f)

src = ''.join(nb['cells'][6]['source'])

keywords = ['sinkhorn', 'hungarian', 'munkres', 'perpassage', 'relative_length', 'latin', 'anchor']
for kw in keywords:
    count = src.lower().count(kw)
    print(f"Keyword '{kw}': {count} occurrences")

