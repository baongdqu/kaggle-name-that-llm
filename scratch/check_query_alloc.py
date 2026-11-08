import json

with open('scratch/lightning_27b/0-637-lora-llama-qwen3-8-27b.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    raw = c.get('source', '')
    src = "".join(raw) if isinstance(raw, list) else raw
    if 'select_low_margin' in src:
        print(f"Cell {i} ({c['cell_type']}):")
        for line in src.splitlines():
            if any(k in line for k in ['def select_low_margin', 'margin', 'threshold', 'quantile', 'top2', 'diff']):
                print("   ", line[:120])
        print("\n--- Full Cell Source ---")
        print(src[:2000])
