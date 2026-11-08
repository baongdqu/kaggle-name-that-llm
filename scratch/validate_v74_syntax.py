import json

with open('champion_v74/champion-v74.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Extract infer.py code from cell 6
infer_py_code = "".join(nb['cells'][6]['source'])
if infer_py_code.startswith("%%writefile infer.py\n"):
    infer_py_code = infer_py_code[len("%%writefile infer.py\n"):]

with open('scratch/test_infer_v74.py', 'w', encoding='utf-8') as f:
    f.write(infer_py_code)

print("test_infer_v74.py written. Validating syntax with py_compile...")
import py_compile
py_compile.compile('scratch/test_infer_v74.py', doraise=True)
print("100% CLEAN SYNTAX! TokAttn integration compiles with zero syntax errors!")

