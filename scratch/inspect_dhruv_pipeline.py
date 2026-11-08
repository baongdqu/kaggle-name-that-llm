import json

with open("scratch/dhruv_repro/name-that-llm-fully-reproducible.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

c9 = "".join(nb["cells"][9]["source"])
print(f"Total lines in Cell 09 (pipeline.py): {len(c9.splitlines())}")
# Print first 50 lines and last 50 lines
lines = c9.splitlines()
print("--- First 30 lines ---")
for l in lines[:30]:
    print(l)
print("\n--- Last 30 lines ---")
for l in lines[-30:]:
    print(l)
