import json, copy

with open('champion_v76/champion-v76.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# V83: The Cleaned Apex Hungarian (Proven V68 architecture, clean of LoRA None ghost reference)
# In Cell 12: Remove q8_sibling_l24 completely
cell12_src = "".join(nb['cells'][12]['source'])
target_line = '  {"name": "q8_sibling_l24", "type": "l24", "path": B8, "adapters": [SIBLING_LORA], "keep": SIBLING_LORAC[\'keep\'], "layers": SIBLING_LORAC[\'layers\'], "w": 0.20, "bs": 16, "max_len": SIBLING_LORAC[\'max_len\'], "twin": 0.35, "dp": True},\n'
if target_line not in cell12_src:
    # Try finding without exact w
    for line in cell12_src.split('\n'):
        if 'q8_sibling_l24' in line:
            cell12_src = cell12_src.replace(line + '\n', '')
else:
    cell12_src = cell12_src.replace(target_line, '')

nb_v83 = copy.deepcopy(nb)
nb_v83['cells'][12]['source'] = [cell12_src]

with open('champion_v83/champion-v83.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_v83, f, indent=2)
print("V83 generated successfully!")

# V84: Qwen3-14B Amplified SOTA
# Amplify Qwen3-14B (w=4.5) + ModernBERT (w=3.0) + DeBERTa (w=1.5), Sinkhorn 0.25 (sharp, clean)
cell12_v84 = cell12_src.replace('"name": "q14v5", "type": "l24", "path": B14, "adapters": AQ14V5, "keep": 27, "layers": L, "w": 3.0',
                                '"name": "q14v5", "type": "l24", "path": B14, "adapters": AQ14V5, "keep": 27, "layers": L, "w": 4.5')
cell12_v84 = cell12_v84.replace('"name": "mb_v5", "type": "enc", "ckpt": MODERNBERT, "w": 2.0',
                                '"name": "mb_v5", "type": "enc", "ckpt": MODERNBERT, "w": 3.0')
cell12_v84 = cell12_v84.replace('"name": "deb", "type": "enc", "ckpt": DEB, "w": 1.0',
                                '"name": "deb", "type": "enc", "ckpt": DEB, "w": 1.5')
cell12_v84 = cell12_v84.replace("'--sinkhorn', '0.45'", "'--sinkhorn', '0.25'")

nb_v84 = copy.deepcopy(nb)
nb_v84['cells'][12]['source'] = [cell12_v84]

with open('champion_v84/champion-v84.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_v84, f, indent=2)
print("V84 generated successfully!")

# V85: Rank Average Diversity Apex
# svc_w=12 (Kchuang breakthrough) + Balanced High-Diversity Weights
cell12_v85 = cell12_src.replace("'--svc_w', '1.5'", "'--svc_w', '8.0'")
cell12_v85 = cell12_v85.replace("'--sinkhorn', '0.45'", "'--sinkhorn', '0.35'")

nb_v85 = copy.deepcopy(nb)
nb_v85['cells'][12]['source'] = [cell12_v85]

with open('champion_v85/champion-v85.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_v85, f, indent=2)
print("V85 generated successfully!")

