import json

with open('champion_v42/champion-v42.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in champion-v42.ipynb: {len(nb['cells'])}")

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        # Strip ipython magics for syntax checking
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

# Check pipeline.py extracted code
pipe_src = ''.join(nb['cells'][9]['source'])
# remove %%writefile pipeline.py
pipe_py = '\n'.join([l for l in pipe_src.split('\n') if not l.startswith('%%')])
compile(pipe_py, 'pipeline.py', 'exec')
print("Extracted pipeline.py syntax: PASS")

# Check comp_stack.py extracted code
comp_src = ''.join(nb['cells'][6]['source'])
comp_py = '\n'.join([l for l in comp_src.split('\n') if not l.startswith('%%')])
compile(comp_py, 'comp_stack.py', 'exec')
print("Extracted comp_stack.py syntax: PASS")

# Check gauss.py extracted code
gauss_src = ''.join(nb['cells'][4]['source'])
gauss_py = '\n'.join([l for l in gauss_src.split('\n') if not l.startswith('%%')])
compile(gauss_py, 'gauss.py', 'exec')
print("Extracted gauss.py syntax: PASS")

print("\nALL 16 CELLS AND MODULES COMPILED WITH ZERO SYNTAX ERRORS!")

