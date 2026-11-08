import json
import ast

nb_path = 'champion_v32/champion-v32.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in V32: {len(nb['cells'])}")
errors = 0

for i, cell in enumerate(nb['cells']):
    c_type = cell['cell_type']
    src = "".join(cell['source'])
    if c_type == 'code':
        code_to_check = []
        for line in src.split('\n'):
            if line.startswith('%') or line.startswith('!'):
                code_to_check.append(f"# {line}")
            else:
                code_to_check.append(line)
        cleaned_code = "\n".join(code_to_check)
        try:
            ast.parse(cleaned_code)
            print(f"Cell {i:2d} (code): OK ({len(src):5d} chars)")
        except SyntaxError as e:
            print(f"Cell {i:2d} (code): SYNTAX ERROR -> {e}")
            errors += 1
    else:
        print(f"Cell {i:2d} (markdown): OK ({len(src):5d} chars)")

# Check pipeline.py specifically
c9_src = "".join(nb['cells'][9]['source'])
if c9_src.startswith('%%writefile pipeline.py'):
    pipeline_code = c9_src.split('%%writefile pipeline.py\n', 1)[1]
    try:
        ast.parse(pipeline_code)
        print("pipeline.py AST parse: SUCCESS (0 syntax errors)")
    except SyntaxError as e:
        print(f"pipeline.py AST parse: SYNTAX ERROR -> {e}")
        errors += 1

# Check key modules
assert 'def topic_correction' in c9_src
assert 'P_corr = topic_correction' in c9_src
print("Topic-Bias De-noising: VERIFIED IN PIPELINE.PY!")

assert 'sibling micro-gates: applied' in c9_src
print("Sibling Micro-Gates: VERIFIED IN PIPELINE.PY!")

assert 'PUBLIC_PREVIEW = len(test_df) == 3' in "".join(nb['cells'][2]['source'])
print("Fast Fixture: VERIFIED IN CELL 2!")

c11_src = "".join(nb['cells'][11]['source'])
assert "w=2.4" in c11_src and "w=1.6" in c11_src
print("Dual Encoders (MB w=2.4, Deb w=1.6): VERIFIED IN CELL 11!")

if errors == 0:
    print("\nALL V32 LINTING AND ARCHITECTURAL CHECKS PASSED WITH 0 ERRORS!")
else:
    print(f"\nFAILED WITH {errors} ERRORS!")
