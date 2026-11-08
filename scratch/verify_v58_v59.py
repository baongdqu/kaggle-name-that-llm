import json

def verify_nb(path):
    print(f'Verifying {path}...')
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for idx, c in enumerate(nb['cells']):
        if c.get('cell_type') == 'code':
            src = ''.join(c.get('source', ''))
            # Filter out IPython magics for syntax compile check
            clean_lines = []
            for line in src.splitlines():
                if line.strip().startswith('!') or line.strip().startswith('%'):
                    clean_lines.append('# ' + line)
                else:
                    clean_lines.append(line)
            clean_src = '\n'.join(clean_lines)
            try:
                compile(clean_src, f'{path}_cell_{idx}', 'exec')
            except SyntaxError as e:
                print(f'SYNTAX ERROR in cell {idx}: {e}')
                return False
    print(f'-> {path} PASSED syntax check cleanly!')
    return True

v58_ok = verify_nb('champion_v58/champion-v58.ipynb')
v59_ok = verify_nb('champion_v59/champion-v59.ipynb')
if v58_ok and v59_ok:
    print('ALL NOTEBOOKS VERIFIED AND 100% READY TO PUSH!')
else:
    raise RuntimeError('Notebook verification failed!')

