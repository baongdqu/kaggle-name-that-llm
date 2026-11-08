import glob, os, re

for root, dirs, files in os.walk('downloaded_kernels'):
    for file in files:
        if file.endswith('.ipynb'):
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            for m in re.finditer(r'[\w\-]+\.safetensors', content):
                print(f"{file}: {m.group(0)}")

