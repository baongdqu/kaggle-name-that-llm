import os, json, shutil, py_compile

print("Creating champion_v92 from champion_v91...", flush=True)

src_dir = 'champion_v91'
dst_dir = 'champion_v92'

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

meta['id'] = 'baonguyenduyquoc/champion-v92'
meta['title'] = 'Champion V92'
meta['code_file'] = 'champion-v92.ipynb'

with open(os.path.join(dst_dir, 'kernel-metadata.json'), 'w') as f:
    json.dump(meta, f, indent=2)

# Load and fix notebook
with open(os.path.join(src_dir, 'champion-v91.ipynb'), 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell_src = nb['cells'][6]['source']
text = cell_src[0] if isinstance(cell_src, list) else cell_src
lines = text.splitlines(keepends=True)

# Fix indentation on line 889: change 8 spaces to 4 spaces
fixed_lines = []
for i, line in enumerate(lines):
    if i == 888 and line.startswith('        _safe = pd.DataFrame'):
        fixed_lines.append('    _safe = pd.DataFrame' + line[len('        _safe = pd.DataFrame'):])
        print(f"Fixed line 889: Changed 8 spaces to 4 spaces: {repr(fixed_lines[-1])}")
    elif "_LAMBDA_WEIGHTS = {'claude_opus': 1.80, 'gpt_56': 0.80, 'gemini_flash': 0.60, 'glm': 0.75}" in line:
        # Upgrade to 72.73% champion weights
        new_line = "    _LAMBDA_WEIGHTS = {'claude_opus': 1.80, 'gpt_56': 1.20, 'gemini_flash': 0.90, 'glm': 22.0}\n"
        fixed_lines.append(new_line)
        print("Upgraded _LAMBDA_WEIGHTS to champion 72.73% configuration!")
    else:
        fixed_lines.append(line)

fixed_text = ''.join(fixed_lines)

# Verify compilation of infer.py
infer_code = fixed_text[fixed_text.find('%%writefile infer.py') + len('%%writefile infer.py\n'):]
with open('scratch/test_compile_infer_v92.py', 'w', encoding='utf-8') as f:
    f.write(infer_code)

try:
    py_compile.compile('scratch/test_compile_infer_v92.py', doraise=True)
    print("✅ VERIFIED: infer.py compiles with ZERO SYNTAX / INDENTATION ERRORS!")
except Exception as e:
    print("❌ COMPILE ERROR:", e)
    raise e

nb['cells'][6]['source'] = [fixed_text]

with open(os.path.join(dst_dir, 'champion-v92.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Successfully generated {dst_dir}/champion-v92.ipynb!")

