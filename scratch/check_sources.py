import json
import subprocess

with open('scratch/public_0639/kernel-metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

print("Checking datasets...")
for ds in meta.get('dataset_sources', []):
    res = subprocess.run(['kaggle', 'datasets', 'files', '-d', ds], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"  [OK] Dataset: {ds}")
    else:
        print(f"  [FAIL] Dataset: {ds} -> {res.stderr.strip()[:100]}")

print("\nChecking kernel sources...")
for ks in meta.get('kernel_sources', []):
    res = subprocess.run(['kaggle', 'kernels', 'output', ks], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"  [OK] Kernel: {ks}")
    else:
        print(f"  [CHECK] Kernel: {ks} (returncode {res.returncode})")

print("\nChecking model sources...")
for ms in meta.get('model_sources', []):
    print(f"  Model: {ms}")
