import json, ast

with open('champion_v37/champion-v37.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = ''.join(c['source']).strip()
        lines = src.splitlines()
        code_lines = []
        for line in lines:
            if line.strip().startswith('%') or line.strip().startswith('!'):
                continue
            code_lines.append(line)
        cleaned_src = '\n'.join(code_lines)
        ast.parse(cleaned_src)

num_cells = len(nb['cells'])
print(f"champion_v37 AST parse PASSED 100%! Total cells: {num_cells}")
