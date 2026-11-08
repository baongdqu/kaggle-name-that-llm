import os, json, shutil, py_compile

print("Creating champion_v93 from champion_v92 with --perpassage 0 (Enabling Hungarian Bipartite Matching on Test Prompts)...", flush=True)

src_dir = 'champion_v92'
dst_dir = 'champion_v93'

if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)
os.makedirs(dst_dir, exist_ok=True)

# Copy support files
for fname in os.listdir(src_dir):
    if fname.endswith(('.py', '.parquet')):
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

# Update kernel-metadata.json
with open(os.path.join(src_dir, 'kernel-metadata.json'), 'r') as f:
    meta = json.load(f)

meta['id'] = 'baonguyenduyquoc/champion-v93'
meta['title'] = 'Champion V93'
meta['code_file'] = 'champion-v93.ipynb'

with open(os.path.join(dst_dir, 'kernel-metadata.json'), 'w') as f:
    json.dump(meta, f, indent=2)

# Load and update notebook
with open(os.path.join(src_dir, 'champion-v92.ipynb'), 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 12: change '--perpassage', '1' to '--perpassage', '0'
for idx, cell in enumerate(nb['cells']):
    src = cell['source']
    new_src = []
    for line in src:
        if "'--perpassage', '1'" in line:
            new_line = line.replace("'--perpassage', '1'", "'--perpassage', '0'")
            new_src.append(new_line)
            print(f"Cell {idx}: Changed '--perpassage', '1' to '--perpassage', '0'!")
        elif '"--perpassage", "1"' in line:
            new_line = line.replace('"--perpassage", "1"', '"--perpassage", "0"')
            new_src.append(new_line)
            print(f"Cell {idx}: Changed '--perpassage', '1' to '--perpassage', '0'!")
        else:
            new_src.append(line)
    cell['source'] = new_src

# Verify infer.py compilation
cell6_src = ''.join(nb['cells'][6]['source'])
infer_code = cell6_src[cell6_src.find('%%writefile infer.py') + len('%%writefile infer.py\n'):]
with open('scratch/test_compile_infer_v93.py', 'w', encoding='utf-8') as f:
    f.write(infer_code)

try:
    py_compile.compile('scratch/test_compile_infer_v93.py', doraise=True)
    print("✅ VERIFIED: infer.py compiles with ZERO SYNTAX / INDENTATION ERRORS!")
except Exception as e:
    print("❌ COMPILE ERROR:", e)
    raise e

with open(os.path.join(dst_dir, 'champion-v93.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Successfully generated {dst_dir}/champion-v93.ipynb!")

