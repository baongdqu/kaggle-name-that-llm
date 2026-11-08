import json

with open('champion_v22/champion-v22.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][6]['source'])
start = src.find(".write_text('") + len(".write_text('")
end = src.rfind("')")
comp_code = src[start:end].encode('utf-8').decode('unicode_escape')

print("comp_code lines:", len(comp_code.splitlines()))
for i, l in enumerate(comp_code.splitlines()):
    if any(k in l for k in ['cuda', 'device', 'ThreadPool', 'ProcessPool', 'parallel', 'torch.no_grad', 'inference_mode']):
        print(f"{i:03d}: {l[:100]}")
