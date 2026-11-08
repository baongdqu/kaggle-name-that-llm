import numpy as np
import torch, json
from safetensors.torch import save_file, load_file

rf = np.load('dataset_sibling_lora/readouts.npz')
norms = load_file('dataset_sibling_lora/norms.safetensors')

heads_dict = {}
for k, v in norms.items():
    heads_dict[k] = v

heads_dict['plain.weight'] = torch.tensor(rf['plain_coef'], dtype=torch.float32)
heads_dict['plain.bias'] = torch.tensor(rf['plain_intercept'], dtype=torch.float32)
heads_dict['centered.weight'] = torch.tensor(rf['center_coef'], dtype=torch.float32)
heads_dict['centered.bias'] = torch.tensor(rf['center_intercept'], dtype=torch.float32)

save_file(heads_dict, 'dataset_sibling_lora/heads.safetensors')
torch.save(heads_dict, 'dataset_sibling_lora/heads.pt')

meta = {
    'labels': [str(x) for x in rf['classes']],
    'cfg': {
        'keep': 24,
        'layers': [2, 4, 6, 8, 12, 16, 20, 24],
        'max_len': 384,
        'eos_id': None
    }
}
with open('dataset_sibling_lora/meta.json', 'w') as mf:
    json.dump(meta, mf, indent=2)

print('Successfully created heads.safetensors, heads.pt, meta.json in dataset_sibling_lora!')
