import json

nb = json.load(open('champion_v88/champion-v88.ipynb', encoding='utf-8'))

# Update cell 0 title
nb['cells'][0]['source'] = ['# Name That LLM Champion V88 5-View Stylometric & Head-Tail Signature Apex\n']

# Update cell 6 (infer.py) with 5th view: Head-Tail signature view
infer_src = ''.join(nb['cells'][6]['source'])

old_views = 'for view, lo, hi, mf, wv in [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("function", 1, 3, 50000, 0.5)]:'
new_views = '''_ht_prep = lambda s_: (" ".join(str(s_).split()[:50]) + " " + " ".join(str(s_).split()[-50:])) if len(str(s_).split()) > 70 else str(s_)
    for view, lo, hi, mf, wv in [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("function", 1, 3, 50000, 0.5), ("head_tail", 3, 5, 100000, 0.6)]:'''

old_prep = 'prep = _shape if view == "shape" else _func if view == "function" else (lambda s_: s_)'
new_prep = 'prep = _ht_prep if view == "head_tail" else _shape if view == "shape" else _func if view == "function" else (lambda s_: s_)'

old_an = 'an = "char" if view == "shape" else "word" if view == "function" else view'
new_an = 'an = "char" if view in ("shape", "head_tail") else "word" if view == "function" else view'

if old_views in infer_src and old_prep in infer_src and old_an in infer_src:
    infer_src = infer_src.replace(old_views, new_views).replace(old_prep, new_prep).replace(old_an, new_an)
    nb['cells'][6]['source'] = [l + '\n' for l in infer_src.splitlines()]
    print("SUCCESS: 5-View Stylometrics (Char + Word + Shape + Function + HeadTail) integrated into infer.py")
else:
    print("WARNING: Views pattern not matched exactly in infer_src!")

# Update cell 12 parameters: svc_w=12.0
cmd_src = ''.join(nb['cells'][12]['source'])
cmd_src = cmd_src.replace("'--svc_w', '10.0'", "'--svc_w', '12.0'")
nb['cells'][12]['source'] = [l + '\n' for l in cmd_src.splitlines()]
print("SUCCESS: Updated --svc_w to 12.0 in Cell 12")

with open('champion_v88/champion-v88.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("champion_v88/champion-v88.ipynb generated successfully!")

