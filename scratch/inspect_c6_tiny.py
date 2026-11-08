import json

with open("scratch/dhruv_repro/name-that-llm-fully-reproducible.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

c6 = "".join(nb["cells"][6]["source"])
lines = c6.splitlines()
print("Total lines in comp_stack.py:", len(lines))
for i in range(75, 120):
    if i < len(lines):
        print(f"L{i:03d}: {lines[i]}")
