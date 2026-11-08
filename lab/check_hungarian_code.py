with open('champion_v61/champion-v61.ipynb', 'r', encoding='utf-8') as f:
    import json
    nb = json.load(f)

src = ''.join(nb['cells'][6]['source'])
lines = src.split('\n')
for i, line in enumerate(lines):
    if 'hungarian' in line.lower() or 'sinkhorn' in line.lower() or 'length' in line.lower() and ('z_' in line or 'beta' in line or 'cluster' in line):
        start = max(0, i - 3)
        end = min(len(lines), i + 4)
        print(f"--- Around line {i} ---")
        for j in range(start, end):
            print(f"{j}: {lines[j]}")
        print()

