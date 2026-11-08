import json

with open("champion_v29/champion-v29.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Patch Cell 2: model_dir_optional
c2_src = "".join(nb["cells"][2]["source"])

old_c2 = """QWEN8, QWEN14 = model_dir('/qwen-3/', '/8b-base/'), model_dir('/qwen-3/', '/14b-base/')
QWEN25_14, LLAMA = model_dir('/qwen2.5/', '/14b/'), model_dir('/llama-3.1/', '/8b/')"""

new_c2 = """def model_dir_opt(*parts):
    hits = sorted(os.path.dirname(c) for c in glob.glob('/kaggle/input/**/config.json', recursive=True) if all(p in c.lower() for p in parts))
    return hits[0] if hits else None

QWEN8, QWEN14 = model_dir('/qwen-3/', '/8b-base/'), model_dir('/qwen-3/', '/14b-base/')
QWEN25_14 = model_dir('/qwen2.5/', '/14b/')
LLAMA = model_dir_opt('/llama-3.1/', '/8b/')"""

assert old_c2 in c2_src, "Failed to match old_c2 in Cell 2!"
c2_src = c2_src.replace(old_c2, new_c2)
nb["cells"][2]["source"] = [line + "\n" for line in c2_src.splitlines()]

# Patch Cell 11: MEMBERS
c11_src = "".join(nb["cells"][11]["source"])

old_c11 = """L = [2, 4, 6, 8, 12, 16, 20, 24]
MEMBERS = [
    dict(name='qwen3_14b', path=QWEN14, adapters=folds('q14v5'), keep=27, layers=L, w=3.0, bs=8,  max_len=384, twin=0.35, gauss=0.70),
    dict(name='qwen3_8b',  path=QWEN8,  adapters=folds('q8v5'),  keep=24, layers=L, w=2.0, bs=16, max_len=384, twin=0.35, gauss=0.44),
    dict(name='llama_8b',  path=LLAMA,  adapters=folds('ll8v5'), keep=24, layers=L, w=1.0, bs=16, max_len=384, twin=0.35),
    dict(name='nishant_14b',   path=QWEN14,    adapters=[N14], keep=N14_CFG['keep'], layers=N14_CFG['layers'], w=2.5, bs=8,
         max_len=N14_CFG['max_len'], twin=0.35, no_final_norm=True, eos_id=N14_CFG['eos_id']),
    dict(name='nishant_q25_14b', path=QWEN25_14, adapters=[Q25], keep=Q25_CFG['keep'], layers=Q25_CFG['layers'], w=1.0, bs=8,
         max_len=Q25_CFG['max_len'], twin=0.35, no_final_norm=True, eos_id=Q25_CFG['eos_id']),
    dict(name='deberta',    type='enc', ckpt=DEBERTA,    w=1.0),
    dict(name='modernbert', type='enc', ckpt=MODERNBERT, w=1.5),
]
BUDGET = int(8.3 * 3600 - STACK_SECONDS - 600)"""

new_c11 = """L = [2, 4, 6, 8, 12, 16, 20, 24]
MEMBERS = [
    dict(name='qwen3_14b', path=QWEN14, adapters=folds('q14v5'), keep=27, layers=L, w=3.0, bs=8,  max_len=384, twin=0.35, gauss=0.70),
    dict(name='qwen3_8b',  path=QWEN8,  adapters=folds('q8v5'),  keep=24, layers=L, w=2.0, bs=16, max_len=384, twin=0.35, gauss=0.44),
    dict(name='nishant_14b',   path=QWEN14,    adapters=[N14], keep=N14_CFG['keep'], layers=N14_CFG['layers'], w=2.5, bs=8,
         max_len=N14_CFG['max_len'], twin=0.35, no_final_norm=True, eos_id=N14_CFG['eos_id']),
    dict(name='nishant_q25_14b', path=QWEN25_14, adapters=[Q25], keep=Q25_CFG['keep'], layers=Q25_CFG['layers'], w=1.0, bs=8,
         max_len=Q25_CFG['max_len'], twin=0.35, no_final_norm=True, eos_id=Q25_CFG['eos_id']),
    dict(name='deberta',    type='enc', ckpt=DEBERTA,    w=1.0),
    dict(name='modernbert', type='enc', ckpt=MODERNBERT, w=1.5),
]
if LLAMA is not None:
    MEMBERS.append(dict(name='llama_8b', path=LLAMA, adapters=folds('ll8v5'), keep=24, layers=L, w=1.0, bs=16, max_len=384, twin=0.35))
    print(f"LLAMA-3.1-8B enabled at {LLAMA}")
else:
    print("LLAMA-3.1-8B not attached or license pending; running 6-member ModernBERT + Two-Stage LoRA24 stack!")

BUDGET = int(8.3 * 3600 - STACK_SECONDS - 600)"""

assert old_c11 in c11_src, "Failed to match old_c11 in Cell 11!"
c11_src = c11_src.replace(old_c11, new_c11)
nb["cells"][11]["source"] = [line + "\n" for line in c11_src.splitlines()]

with open("champion_v29/champion-v29.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("SUCCESS: champion_v29/champion-v29.ipynb patched successfully!")
