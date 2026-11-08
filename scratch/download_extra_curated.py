import subprocess
import os
import shutil
import glob

base_dir = os.path.abspath("downloaded_datasets")
os.makedirs(base_dir, exist_ok=True)

extra_datasets = [
    ("08_train_prompts_reconstructed", "pjmathematician/ntl-train-prompts-reconstructed"),
    ("09_same_prompt_high_fidelity", "haileycheng/name-that-llm-same-prompt-passages"),
    ("10_openrouter_harvest", "willrice/name-that-llm-openrouter-harvest"),
    ("11_curated_pool_routed_86k", "ankiteshtiwari/ntl-curated-pool-routed"),
]

for folder_name, slug in extra_datasets:
    target_dir = os.path.join(base_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    print(f"\n=== Downloading {folder_name} ({slug}) ===")
    res = subprocess.run(["kaggle", "datasets", "download", slug, "-p", target_dir, "--unzip"], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error downloading {slug}: {res.stderr}")
    else:
        print(f"Successfully downloaded & unzipped {folder_name}")

print("\n=== Summary of Newly Downloaded Extra Curated Datasets ===")
for folder_name, _ in extra_datasets:
    td = os.path.join(base_dir, folder_name)
    for f in os.listdir(td):
        fp = os.path.join(td, f)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp) / 1024.0
            print(f"{folder_name}/{f:35s} | {sz:8.1f} KB")

