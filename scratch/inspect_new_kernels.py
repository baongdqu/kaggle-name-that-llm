import json

for path, name in [
    ("scratch/dhruv_repro/name-that-llm-fully-reproducible.ipynb", "Dhruv Repro"),
    ("scratch/drozanes_apex/name-that-llm-apex.ipynb", "Drozanes Apex")
]:
    print(f"=== {name} ===")
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        for i, c in enumerate(nb["cells"][:5]):
            src = "".join(c.get("source", []))
            print(f"Cell {i} [{c.get('cell_type')}]: {src.strip()[:150]}")
        # check last 3 cells
        print("--- Tail cells ---")
        for i, c in enumerate(nb["cells"][-3:], start=len(nb["cells"])-3):
            src = "".join(c.get("source", []))
            print(f"Cell {i} [{c.get('cell_type')}]: {src.strip()[:150]}")
    except Exception as e:
        print("Error:", e)
