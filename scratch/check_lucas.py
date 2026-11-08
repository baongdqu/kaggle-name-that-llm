import json

def inspect_nb(path):
    print(f"=== {path} ===")
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for i, c in enumerate(nb['cells']):
        text = ''.join(c.get('source', []))
        if any(k in text.lower() for k in ['trainer', 'modernbert', 'loss', 'safetensors', 'encoder', 'head.pt']):
            print(f"Cell {i} ({c['cell_type']}):")
            print(text[:500])
            print("---")

inspect_nb('scratch/lucas_enc/name-that-llm-extract-encoders2.ipynb')

