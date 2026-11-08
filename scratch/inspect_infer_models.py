import json

with open('champion_v68/champion-v68.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

infer_src = "".join(nb['cells'][6]['source'])

keywords = ['glm4', 'electra', 'tokattn', 'models =', 'checkpoints', 'load_model', 'AutoModel']
for line in infer_src.split('\n'):
    if any(k in line.lower() for k in ['qwen', 'llama', 'modernbert', 'deberta', 'pipeline', 'weights', 'torch.load', 'state_dict']):
        print(line[:120])

