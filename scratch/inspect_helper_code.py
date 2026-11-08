with open("scratch/comp_stack_dump.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(805, min(840, len(lines))):
    print(f"L{i+1:04d}: {lines[i]}", end="")
