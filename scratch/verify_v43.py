import json

with open('champion_v43/champion-v43.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in champion-v43.ipynb: {len(nb['cells'])}")

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        clean_lines = []
        for line in src.split('\n'):
            if line.startswith('%') or line.startswith('!'):
                clean_lines.append('# ' + line)
            else:
                clean_lines.append(line)
        clean_src = '\n'.join(clean_lines)
        try:
            compile(clean_src, f'cell_{i}', 'exec')
            print(f"Cell {i:2d} syntax: PASS")
        except SyntaxError as e:
            print(f"Cell {i:2d} syntax: FAIL -> {e}")
            raise e

# Verify pipeline.py
pipe_src = ''.join(nb['cells'][9]['source'])
pipe_py = '\n'.join([l for l in pipe_src.split('\n') if not l.startswith('%%')])
compile(pipe_py, 'pipeline.py', 'exec')
print("Extracted pipeline.py syntax: PASS")

# Verify comp_stack.py
comp_src = ''.join(nb['cells'][6]['source'])
comp_py = '\n'.join([l for l in comp_src.split('\n') if not l.startswith('%%')])
compile(comp_py, 'comp_stack.py', 'exec')
print("Extracted comp_stack.py syntax: PASS")

# Verify gauss.py
gauss_src = ''.join(nb['cells'][4]['source'])
gauss_py = '\n'.join([l for l in gauss_src.split('\n') if not l.startswith('%%')])
compile(gauss_py, 'gauss.py', 'exec')
print("Extracted gauss.py syntax: PASS")

print("\nALL 16 CELLS AND MODULES IN V43 COMPILED WITH ZERO SYNTAX ERRORS!")

