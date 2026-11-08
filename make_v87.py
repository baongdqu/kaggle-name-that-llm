import json

nb = json.load(open('champion_v87/champion-v87.ipynb', encoding='utf-8'))

# Update cell 0 title
nb['cells'][0]['source'] = ['# Name That LLM Champion V87 High-Coverage Sibling Disentanglement Apex\n']

# Update cell 6 (infer.py) with amplified lambda weights
infer_src = ''.join(nb['cells'][6]['source'])
old_lam = "_LAMBDA_WEIGHTS = {'claude_opus': 1.0, 'gpt_56': 0.40, 'gemini_flash': 0.25}"
new_lam = "_LAMBDA_WEIGHTS = {'claude_opus': 1.20, 'gpt_56': 0.75, 'gemini_flash': 0.50}"

if old_lam in infer_src:
    infer_src = infer_src.replace(old_lam, new_lam)
    nb['cells'][6]['source'] = [l + '\n' for l in infer_src.splitlines()]
    print("SUCCESS: Updated _LAMBDA_WEIGHTS in infer.py")
else:
    print("WARNING: old_lam not found in infer_src!")

# Update cell 12 (args) with adapt_thr=0.03, svc_w=10.0
cmd_src = ''.join(nb['cells'][12]['source'])
cmd_src = cmd_src.replace("'--adapt_thr', '0.15'", "'--adapt_thr', '0.03'")
cmd_src = cmd_src.replace("'--svc_w', '8.0'", "'--svc_w', '10.0'")
nb['cells'][12]['source'] = [l + '\n' for l in cmd_src.splitlines()]
print("SUCCESS: Updated --adapt_thr to 0.03 and --svc_w to 10.0 in Cell 12")

with open('champion_v87/champion-v87.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("champion_v87/champion-v87.ipynb generated successfully!")

