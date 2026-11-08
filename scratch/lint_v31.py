import json
import ast
import re

nb_path = 'champion_v31/champion-v31.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in V31: {len(nb['cells'])}")
errors = 0

for i, cell in enumerate(nb['cells']):
    c_type = cell['cell_type']
    src = "".join(cell['source'])
    if c_type == 'code':
        # Remove ipython magics for ast parsing
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

# Check comp_stack.py specifically
c6_src = "".join(nb['cells'][6]['source'])
if c6_src.startswith('%%writefile comp_stack.py'):
    comp_code = c6_src.split('%%writefile comp_stack.py\n', 1)[1]
    try:
        ast.parse(comp_code)
        print("comp_stack.py AST parse: SUCCESS (0 syntax errors)")
    except SyntaxError as e:
        print(f"comp_stack.py AST parse: SYNTAX ERROR -> {e}")
        errors += 1

# Check Fast Fixture condition
assert 'PUBLIC_PREVIEW = len(test_df) == 3' in "".join(nb['cells'][2]['source'])
assert 'if PUBLIC_PREVIEW:' in "".join(nb['cells'][7]['source'])
assert 'if PUBLIC_PREVIEW:' in "".join(nb['cells'][13]['source'])
print("Fast Fixture presence: VERIFIED IN CELLS 2, 7, 13!")

# Check Sibling Micro-Gates
assert 'sibling micro-gates: applied' in c9_src
print("Sibling Micro-Gates presence: VERIFIED IN PIPELINE.PY!")

# Check MEMBERS weights
c11_src = "".join(nb['cells'][11]['source'])
assert "w=2.4" in c11_src and "w=1.6" in c11_src
print("ModernBERT w=2.4 & DeBERTa w=1.6: VERIFIED IN CELL 11!")

# Check comp_w
c13_src = "".join(nb['cells'][13]['source'])
assert "'--comp_w', '0.22'" in c13_src
print("comp_w = 0.22: VERIFIED IN CELL 13!")

if errors == 0:
    print("\nALL V31 LINTING AND ARCHITECTURE CHECKS PASSED PERFECTLY!")
else:
    print(f"\nFAILED WITH {errors} ERRORS!")
