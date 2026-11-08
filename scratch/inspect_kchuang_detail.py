import json

with open('scratch/kchuang_0668/ntl-0668-stack-pool-prompt-rows.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = "".join(c['source'])
        first_line = c['source'][0].strip() if c['source'] else ""
        print(f"Cell {i} (len {len(c['source'])}): {first_line[:80]}")
        # search for pool, stack, merge, override, submission
        matches = [line.strip() for line in c['source'] if any(k in line.lower() for k in ['open_pool', 'stack', 'override', 'submission', 'merge', 'join', 'prompt_id', 'exact'])]
        if matches:
            print(f"   Matches: {matches[:3]}")

