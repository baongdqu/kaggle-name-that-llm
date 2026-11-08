import py_compile

# Test compiling worker_code directly from python
from generate_sibling_training_nb import worker_code

with open('scratch/test_worker_direct.py', 'w', encoding='utf-8') as f:
    f.write(worker_code)

py_compile.compile('scratch/test_worker_direct.py', doraise=True)
print("Worker code verified: 100% CLEAN SYNTAX!")

