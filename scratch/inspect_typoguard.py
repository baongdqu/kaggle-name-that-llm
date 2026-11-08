import json

def check(nb_path):
    print(f"\n=================== {nb_path} ===================")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for i, c in enumerate(nb['cells']):
        raw = c.get('source', '')
        src = "".join(raw) if isinstance(raw, list) else raw
        if any(w in src.lower() for w in ['typoguard', 'typo', 'quote', 'dash', 'ratio', 'sinkhorn', 'blend', 'guard']):
            print(f"\n--- Cell {i} ({c['cell_type']}) ---")
            for line in src.splitlines():
                if any(w in line.lower() for w in ['typoguard', 'typo', 'quote', 'dash', 'guard', 'ratio', 'weight', '0.6']):
                    print("  ", line[:120])

check('scratch/typoguard_nb/0-624-two-stage-lora-encoders-typoguard.ipynb')
check('scratch/picklerick_nb/ntl-0-627-encoder-ratio-6-to-1.ipynb')
