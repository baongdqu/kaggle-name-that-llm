import json

with open(r'scratch/pub_0673/ntl-we-back-0-673.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source'])
    cell_type = cell['cell_type']
    if cell_type == 'markdown':
        print(f"\n--- [Cell {i}] Markdown ---")
        print(source[:300])
    elif cell_type == 'code':
        # Look for model names, weights, Hungarian, SVC, TFIDF
        keywords = ['weight', 'blend', 'hungarian', 'linear_sum_assignment', 'Tfidf', 'SVC', 'sinkhorn', 'probs', 'ensemble', '0.67']
        if any(k.lower() in source.lower() for k in keywords):
            print(f"\n--- [Cell {i}] Code snippet ---")
            for line in source.split('\n'):
                if any(k.lower() in line.lower() for k in keywords):
                    print("  ", line[:120])

