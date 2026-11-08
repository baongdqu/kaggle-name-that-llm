import json

with open("champion_v30/champion-v30.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
assert len(nb["cells"]) == 16, f"Expected 16 cells, got {len(nb['cells'])}"

# Compile every code cell
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        lines = [l for l in src.splitlines() if not l.startswith("%%") and not l.startswith("!")]
        code_to_check = "\n".join(lines)
        try:
            compile(code_to_check, f"cell_{i}", "exec")
        except SyntaxError as e:
            print(f"SyntaxError in cell {i}: {e}")
            raise
        print(f"Cell {i} compiles cleanly ({len(lines)} lines)")

print("ALL CODE CELLS COMPILED WITH 0 SYNTAX ERRORS!")

# Verify metadata
with open("champion_v30/kernel-metadata.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

print(f"Kernel ID: {meta['id']}")
print(f"Code file: {meta['code_file']}")
print(f"Datasets attached: {len(meta['dataset_sources'])}")
print(f"Kernels attached: {len(meta['kernel_sources'])}")
print(f"Models attached: {len(meta['model_sources'])}")

assert meta['id'] == 'baonguyenduyquoc/name-that-llm-champion-v30-apex-frontier-hybrid'
assert 'metaresearch/llama-3.1/Transformers/8b/2' not in meta['model_sources']
print("CONFIRMED: Clean metadata ready for 100% successful push and fast commit!")
