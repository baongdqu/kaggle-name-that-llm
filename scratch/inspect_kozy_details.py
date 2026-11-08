import json

with open('scratch/kozykappa/name-that-llm-enhanced-0-651-recipe.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c3_src = "".join(nb['cells'][3]['source'])
print("=== ntl_upgrade.py ===")
print(c3_src[:2000])

c6_src = "".join(nb['cells'][6]['source'])
print("\n=== infer.py ===")
lines = c6_src.split('\n')
for l in lines:
    if any(k in l for k in ['TOPIC_ALPHA', 'topic_alpha', 'correction', 'svc', 'alpha']):
        print("  ->", l.strip()[:120])
