with open("scratch/comp_stack_dump.py", "r", encoding="utf-8") as f:
    text = f.read()

lines = text.splitlines()
print(f"Total lines in comp_stack_dump.py: {len(lines)}")
for idx, l in enumerate(lines):
    if l.strip().startswith("#") and any(k in l.lower() for k in ["===", "model", "step", "weight", "member", "stack"]):
        print(f"L{idx:04d}: {l.strip()[:100]}")
    elif "night24_helper" in l:
        print(f"L{idx:04d}: {l.strip()[:100]}")
