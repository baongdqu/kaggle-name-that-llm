import json

def read_cells(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    cells = []
    for c in nb['cells']:
        text = ''.join(c.get('source', []))
        cells.append((c.get('cell_type'), text))
    return cells

# 1. Alex Chilton: What failed and what worked
ac = read_cells('scratch/intel/alex_chilton/name-that-llm-0-662-all-public-what-failed.ipynb')
print("=== ALEX CHILTON: WHAT FAILED / WHAT WORKED ===")
for ctype, text in ac:
    if any(k in text.lower() for k in ['what we tried', 'did not work', 'what failed', 'what worked', 'public score']):
        print(text[:1500])
        print("--------------------------------------------------")

# 2. Error Map: Top confused pairs
em = read_cells('scratch/intel/error_map/name-that-llm-error-map-of-the-18-models.ipynb')
print("\n=== ERROR MAP: CONFUSED PAIRS & RECALL ===")
for ctype, text in em:
    if any(k in text.lower() for k in ['most confused', 'per-class recall', 'wrong family', 'confusion map']):
        print(text[:1500])
        print("--------------------------------------------------")

