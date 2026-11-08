import json
import py_compile

with open('champion_v47/champion-v47.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
errors = 0

for i, cell in enumerate(nb['cells']):
    ctype = cell.get('cell_type')
    src = ''.join(cell.get('source', []))
    if ctype == 'code':
        # Check if cell has %%writefile
        lines = src.split('\n')
        if lines and lines[0].startswith('%%writefile'):
            fname = lines[0].split()[1]
            code = '\n'.join(lines[1:])
            try:
                compile(code, fname, 'exec')
                print(f"Cell {i} ({fname}): COMPILE OK ({len(lines)} lines)")
            except Exception as e:
                print(f"Cell {i} ({fname}) COMPILE ERROR: {e}")
                errors += 1
        else:
            # Normal code cell (strip ipython magics if any like !python)
            clean_lines = [l for l in lines if not l.strip().startswith('!') and not l.strip().startswith('%')]
            code = '\n'.join(clean_lines)
            try:
                compile(code, f"cell_{i}.py", 'exec')
                print(f"Cell {i} (inline code): COMPILE OK ({len(lines)} lines)")
            except Exception as e:
                print(f"Cell {i} COMPILE ERROR: {e}")
                errors += 1

if errors == 0:
    print("ALL CELLS COMPILED CLEANLY WITH ZERO ERRORS!")
else:
    print(f"FOUND {errors} COMPILE ERRORS!")

