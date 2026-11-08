import json

nb = json.load(open('champion_v33/champion-v33.ipynb', encoding='utf-8'))
for i in [6, 7, 9, 11, 13, 14]:
    source = "".join(nb['cells'][i]['source'])
    print(f"=== CELL {i} (len {len(source)}) ===")
    lines = source.split('\n')
    for line in lines[:20]:
        print("  ", line)
    if len(lines) > 20:
        print("   ... (truncated)")
