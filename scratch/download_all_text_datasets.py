import subprocess
import os
import shutil
import glob
import pandas as pd

base_dir = os.path.abspath("downloaded_datasets")
os.makedirs(base_dir, exist_ok=True)

# 1. Download Competition Data
comp_dir = os.path.join(base_dir, "00_competition_data")
os.makedirs(comp_dir, exist_ok=True)
print("=== Downloading 00_competition_data (name-that-llm) ===")
subprocess.run(["kaggle", "competitions", "download", "-c", "name-that-llm", "-p", comp_dir], check=False)
for zip_file in glob.glob(os.path.join(comp_dir, "*.zip")):
    shutil.unpack_archive(zip_file, comp_dir)
    os.remove(zip_file)

# Text / Tabular datasets to download
datasets = [
    ("01_open_pool_198k", "crtproduction/name-that-llm-open-pool"),
    ("02_nlm_matched_clean", "dhruvtibarewal/nlm-matched-clean"),
    ("03_extra_api_passages", "seyitkaangunes/ntl-extra-api-passages"),
    ("04_ntl_generations", "abdullah1afify/ntl-generations"),
    ("05_proxy_generations", "dophuchung/ntl-proxy-generations"),
    ("06_free_proxy_generations", "lucasboesen/ntl-free-proxy-generations"),
    ("07_extended_dataset", "prvsiyan/prvsiyans-extended-name-that-llm-dataset"),
]

for folder_name, slug in datasets:
    target_dir = os.path.join(base_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    print(f"\n=== Downloading {folder_name} ({slug}) ===")
    res = subprocess.run(["kaggle", "datasets", "download", slug, "-p", target_dir, "--unzip"], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error downloading {slug}: {res.stderr}")
    else:
        print(f"Successfully downloaded & unzipped {folder_name}")

print("\n=== Dataset Download Summary & Inventory ===")
summary = []
for root, dirs, files in os.walk(base_dir):
    for f in files:
        fpath = os.path.join(root, f)
        relpath = os.path.relpath(fpath, base_dir)
        size_kb = os.path.getsize(fpath) / 1024.0
        row_count = "N/A"
        try:
            if f.endswith('.csv'):
                # quick line count
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as fp:
                    row_count = sum(1 for _ in fp) - 1
            elif f.endswith('.parquet'):
                df = pd.read_parquet(fpath)
                row_count = len(df)
        except Exception as e:
            row_count = f"Err: {e}"
        summary.append((relpath, f"{size_kb:.1f} KB", row_count))

for s in summary:
    print(f"{s[0]:60s} | {s[1]:>12s} | Rows: {s[2]}")

