import json, subprocess

# Test if kaggle allows attaching google/gemma-2/Transformers/9b/2
meta = {
  "id": "baonguyenduyquoc/test-gemma2-model-source",
  "title": "Test Gemma2 Model Source",
  "code_file": "test.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": True,
  "enable_gpu": False,
  "enable_tpu": False,
  "enable_internet": False,
  "dataset_sources": [
    "alexchilton/ntl-ts-gemma2-9b"
  ],
  "model_sources": [
    "google/gemma-2/Transformers/9b/2"
  ]
}

import os
os.makedirs("scratch/test_gemma", exist_ok=True)
with open("scratch/test_gemma/kernel-metadata.json", "w") as f:
    json.dump(meta, f)
with open("scratch/test_gemma/test.py", "w") as f:
    f.write("print('hello gemma')\n")

res = subprocess.run(["kaggle", "kernels", "push", "-p", "scratch/test_gemma"], capture_output=True, text=True)
print("Returncode:", res.returncode)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
