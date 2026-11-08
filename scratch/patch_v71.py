import json

with open('champion_v71/champion-v71.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update title
nb['cells'][0]['source'] = ['# Name That LLM Champion V71 Triad MoS Atlas Apex (Targeting 0.675+ Breakthrough)\n']

src = ''.join(nb['cells'][6]['source'])

old_targets = """        _mos_targets = {
            'gemini_flash': [LABELS.index('gemini-3.6-flash'), LABELS.index('gemini-3.7-flash'), LABELS.index('gemini-3.8-flash')],
            'glm53': [LABELS.index('glm-5.3'), LABELS.index('glm-5.3-flash')]
        }"""

new_targets = """        _mos_targets = {
            'gemini_flash': [LABELS.index('gemini-3.6-flash'), LABELS.index('gemini-3.7-flash'), LABELS.index('gemini-3.8-flash')],
            'glm53': [LABELS.index('glm-5.3'), LABELS.index('glm-5.3-flash')],
            'gpt56': [LABELS.index('gpt-5.6-luna'), LABELS.index('gpt-5.6-sol'), LABELS.index('gpt-5.6-terra')]
        }"""

assert old_targets in src, "old_targets not found in cell 6!"
src = src.replace(old_targets, new_targets)
nb['cells'][6]['source'] = [src]

# Cell 12: Boost atlas_w to 0.25 and atlas_margin to 0.35
c12 = ''.join(nb['cells'][12]['source'])
c12 = c12.replace("'--atlas_w', '0.20'", "'--atlas_w', '0.25'")
c12 = c12.replace("'--atlas_margin', '0.3'", "'--atlas_margin', '0.35'")
nb['cells'][12]['source'] = [c12]

with open('champion_v71/champion-v71.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated champion-v71.ipynb successfully!')

