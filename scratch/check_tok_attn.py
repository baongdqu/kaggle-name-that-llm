import json

with open('champion_v47/champion-v47.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

comp_src = ''.join(nb['cells'][8]['source'])
idx = comp_src.find('token_attention_probs')
while idx != -1:
    print(comp_src[idx-50:idx+250])
    print("="*50)
    idx = comp_src.find('token_attention_probs', idx+1)

