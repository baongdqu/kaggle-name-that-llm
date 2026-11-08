import json

def inspect(path):
    print(f"==================== {path} ====================")
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print("Num cells:", len(nb['cells']))
    for i, c in enumerate(nb['cells']):
        src = "".join(c.get('source', []))
        if any(w in src.lower() for w in ['model', 'load', 'output', 'save', 'weights', 'score', 'acc', 'electra', 'glm', 'train']):
            print(f"--- Cell {i} ({c['cell_type']}) ---")
            print(src[:600])
            print()

inspect("scratch/lucas_glm/name-that-llm-glm4-9b-trunc25.ipynb")
inspect("scratch/lucas_electra/name-that-llm-electra-large-weights.ipynb")
inspect("scratch/lucas_tokattn/name-that-llm-train-tokattn2.ipynb")

