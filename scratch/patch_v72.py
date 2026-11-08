import json

with open('champion_v72/champion-v72.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update title
nb['cells'][0]['source'] = ['# Name That LLM Champion V72 Per-Prompt Relative MoS Apex (Targeting 0.675+ Breakthrough)\n']

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

# Adjust w_spec in cell 6 to 0.012 for slightly stronger signal
src = src.replace('_w_spec = 0.008', '_w_spec = 0.012')
nb['cells'][6]['source'] = [src]

with open('champion_v72/champion-v72.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated champion-v72.ipynb successfully!')

