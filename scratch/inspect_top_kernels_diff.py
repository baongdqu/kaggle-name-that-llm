import json
import glob
import os

def inspect_nb(path, name):
    print(f"\n==================== {name} ({path}) ====================")
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"Total cells: {len(nb['cells'])}")
    for i, c in enumerate(nb['cells']):
        src = "".join(c['source'])
        # Look for model configs, weights, changes
        if any(k in src for k in ['MEMBERS', 'pipeline.py', 'comp_w', 'atlas_w', 'MODERNBERT', '0.6', 'Sinkhorn', 'blend', 'Qwen3.8-27B']):
            first_line = c['source'][0].strip() if c['source'] else ''
            print(f"  Cell {i:2d} ({c['cell_type']}): {first_line[:90]}")
            if 'MEMBERS' in src or 'comp_w' in src or '0.6' in src:
                for line in c['source']:
                    if any(k in line for k in ['MEMBERS =', 'w=', 'comp_w', 'atlas_w', 'blend', 'weight']):
                        print(f"    -> {line.strip()[:100]}")

inspect_nb('scratch/kozykappa/name-that-llm-enhanced-0-651-recipe.ipynb', 'KOZYKAPPA')
inspect_nb('scratch/lightning_27b_spectral/qwen3-8-27b-lora-spectral-ensemble.ipynb', 'LIGHTNING_27B')
inspect_nb('scratch/areebbb/0-643-same-stack-sharper-blend-enjoy.ipynb', 'AREEBBB')
