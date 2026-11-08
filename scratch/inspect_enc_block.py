import json

with open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find("spec.get('type') == 'enc'")
print(text[idx-50:idx+1200])

