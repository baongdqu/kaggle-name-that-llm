import os
import shutil
import zipfile

target_dir = r"C:\Users\s3cr3t\Downloads\kaggle_ntl_AI_READY"
zip_path = r"C:\Users\s3cr3t\Downloads\kaggle_ntl_AI_READY.zip"

md_files = [
    "AGENTS.md",
    "abc CHAT_HISTORY.md",
    "abc LAB.md",
    "abc SUBMISSIONS.md",
    "abc SCORE_CEILING_AND_BAYES_LIMIT.md",
    "abc LEADERBOARD.md",
    "abc DISCUSSIONS.md"
]

for md in md_files:
    if os.path.exists(md):
        dest_name = os.path.splitext(md)[0] + ".txt"
        dest_path = os.path.join(target_dir, dest_name)
        shutil.copy2(md, dest_path)
        print(f"Synced {md} -> {dest_path}")

# Re-zip
print("Rebuilding zip...")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, target_dir)
            zipf.write(full_path, arcname)

size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"Zip created successfully at {zip_path}: {size_mb:.2f} MB")

