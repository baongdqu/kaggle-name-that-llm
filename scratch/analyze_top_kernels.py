import json
from pathlib import Path

notebooks = [
    ("prvsiyan", "scratch/prvsiyan/the-prompt-is-not-the-author.ipynb"),
    ("lightningv08", "scratch/lightningv08/0-637-lora-llama-qwen3-8-27b.ipynb"),
    ("romantamrazov", "scratch/romantamrazov/name-that-llm-who-am-i.ipynb"),
    ("maverickss26", "scratch/maverickss26/kaggle-name-the-llm-v2.ipynb"),
]

for name, path in notebooks:
    print(f"\n========================================================")
    print(f"=== NOTEBOOK: {name} ({path}) ===")
    print(f"========================================================")
    if not Path(path).exists():
        print("Not found!")
        continue
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"Total cells: {len(nb['cells'])}")
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'markdown':
            src = ''.join(c['source']).strip()
            if any(k in src.lower() for k in ['cv', 'lb', '0.', 'score', 'model', 'lora', 'author', 'finding', 'insight', 'sinkhorn', 'leak', 'prompt']):
                print(f"\n--- Cell {i} (markdown) ---")
                for line in src.splitlines()[:15]:
                    print(line[:120])
