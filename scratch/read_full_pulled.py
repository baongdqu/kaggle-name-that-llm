import json

def full_read(path):
    print(f"==================== {path} ====================")
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for i, c in enumerate(nb['cells']):
        print(f"Cell {i} ({c['cell_type']}):")
        print("".join(c.get('source', [])))
        print("-" * 50)

full_read("scratch/lucas_glm/name-that-llm-glm4-9b-trunc25.ipynb")
full_read("scratch/lucas_electra/name-that-llm-electra-large-weights.ipynb")
full_read("scratch/lucas_tokattn/name-that-llm-train-tokattn2.ipynb")

