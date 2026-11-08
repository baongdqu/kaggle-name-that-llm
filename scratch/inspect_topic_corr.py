import json

with open('scratch/kozykappa/name-that-llm-enhanced-0-651-recipe.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c3_src = "".join(nb['cells'][3]['source'])
idx = c3_src.find('def topic_correction')
if idx != -1:
    print(c3_src[idx:idx+1500])
else:
    print("topic_correction not found!")
