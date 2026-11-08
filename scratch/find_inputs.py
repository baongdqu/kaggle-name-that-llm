import json
import re

with open('champion_v53/champion-v53.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

kaggle_inputs = set()
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        text = ''.join(cell.get('source', []))
        matches = re.findall(r'[\'"]/kaggle/input/[^\'"]+[\'"]', text)
        for m in matches:
            clean = m.strip('\'"')
            kaggle_inputs.add(clean)

print("=== Kaggle Inputs referenced in V53 Code ===")
for k in sorted(kaggle_inputs):
    print(k)

