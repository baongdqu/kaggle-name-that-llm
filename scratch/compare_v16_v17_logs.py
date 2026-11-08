import json
import re

def parse_log(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    # Find all stdout data lines
    stdout_lines = []
    for m in re.finditer(r'\{"stream_name":"stdout",.*?"data":"(.*?)"\}', content):
        data = m.group(1).encode().decode('unicode_escape')
        stdout_lines.extend(data.splitlines())
    return stdout_lines

v16_lines = parse_log('v16_output/name-that-llm-champion-v16-apex-sovereign.log')
v17_lines = parse_log('v17_output/name-that-llm-champion-v17-apex-crown.log')

print(f"V16 stdout lines: {len(v16_lines)}")
print(f"V17 stdout lines: {len(v17_lines)}")

print("\n--- V16 Weight additions and models ---")
for l in v16_lines:
    if any(k in l for k in ['weight', 'lora', 'Scoring', 'Sinkhorn', 'Hungarian', 'cluster', 'modified', 'parity', 'checkpoint']):
        print("  [V16]", l.strip())

print("\n--- V17 Weight additions and models ---")
for l in v17_lines:
    if any(k in l for k in ['weight', 'lora', 'Scoring', 'Sinkhorn', 'Hungarian', 'cluster', 'modified', 'parity', 'checkpoint']):
        print("  [V17]", l.strip())
