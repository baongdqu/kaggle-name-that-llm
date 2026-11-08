import json

with open('champion_v30/champion-v30.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

lines = nb['cells'][9]['source']
for i in range(0, len(lines), 30):
    chunk = "".join(lines[i:i+30])
    print(f"--- Lines {i} to {i+30} ---")
    print(chunk[:500])
