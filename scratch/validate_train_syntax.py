import py_compile
import json

with open('train_sibling_lora24/train-sibling-contrastive-lora24.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Extract python source from cell 1
src = "".join(nb['cells'][1]['source'])
with open('scratch/test_train_syntax.py', 'w', encoding='utf-8') as f:
    f.write(src)

print("Checking test_train_syntax.py...")
py_compile.compile('scratch/test_train_syntax.py', doraise=True)
print("Notebook outer script compiles successfully!")

# Extract worker code from the string literal
idx1 = src.find("WORKER = r'''")
if idx1 == -1:
    idx1 = src.find("WORKER = '''")
if idx1 != -1:
    idx2 = src.find("'''", idx1 + 12)
    worker_src = src[idx1+12:idx2]
    with open('scratch/test_worker_syntax.py', 'w', encoding='utf-8') as f:
        f.write(worker_src)
    print("Checking test_worker_syntax.py...")
    py_compile.compile('scratch/test_worker_syntax.py', doraise=True)
    print("Worker code compiles 100% cleanly!")

