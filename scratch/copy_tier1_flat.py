import os
import shutil
import zipfile

def copy_tier1_flat():
    src_dir = os.path.abspath('.')
    downloads_dir = os.path.expanduser('~/Downloads')
    dest_dir = os.path.join(downloads_dir, 'kaggle_ntl_AI_READY')
    dest_zip = os.path.join(downloads_dir, 'kaggle_ntl_AI_READY.zip')

    # Recreate destination directory
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    allowed_exts = {'.md', '.ipynb', '.py', '.txt', '.yaml', '.yml'}
    skip_dirs = {'.git', '__pycache__', '.venv', 'venv', '.idea', '.vscode', 'node_modules', '.gemini'}
    
    copied_files = []
    skipped_files = []
    total_bytes = 0

    print(f'Starting scan of: {src_dir}')
    print(f'Destination directory: {dest_dir}')

    for root, dirs, files in os.walk(src_dir):
        # Filter out skipped directories in-place
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for f in files:
            src_path = os.path.join(root, f)
            rel_path = os.path.relpath(src_path, src_dir)
            ext = os.path.splitext(f)[1].lower()
            
            try:
                sz = os.path.getsize(src_path)
            except Exception:
                continue

            # Inclusion rules
            should_include = False
            if ext in allowed_exts:
                should_include = True
            elif ext == '.json':
                # Include config and metadata json files (< 5 MB)
                if sz < 5 * 1024 * 1024:
                    should_include = True
            elif ext == '.csv':
                # Include small reference CSVs (< 3 MB, e.g. train.csv, test.csv, prompts.csv)
                if sz < 3 * 1024 * 1024:
                    should_include = True

            if should_include:
                # Construct flattened filename
                # e.g. champion_v59/champion-v59.ipynb -> champion_v59__champion-v59.ipynb
                flat_name = rel_path.replace(os.sep, '__').replace('/', '__').replace('\\', '__')
                
                # Clean up any leading dots or underscores
                flat_name = flat_name.lstrip('.').lstrip('_')
                if not flat_name:
                    flat_name = f
                
                dest_path = os.path.join(dest_dir, flat_name)
                shutil.copy2(src_path, dest_path)
                copied_files.append((flat_name, sz, rel_path))
                total_bytes += sz
            else:
                skipped_files.append((rel_path, sz, ext))

    print(f'Successfully copied {len(copied_files)} files ({total_bytes / (1024**2):.2f} MB) into flat folder.')

    # Create the master AI Guide & Index file
    index_content = f"""# KAGGLE NAME THAT LLM - AI CONTEXT PACK & PROJECT INDEX

> **Generated for external AI assistant consultation (ChatGPT, Claude, Gemini, DeepSeek)**
> **Competition:** Kaggle Name That LLM (Identify the frontier LLM behind the generated text among 18 classes)
> **Evaluation Metric:** Macro-Averaged Recall (Balanced Accuracy across 18 classes)
> **Team:** private / Mohamed Morsi (baonguyenduyquoc) | Team ID: 17137218
> **Current Team Standing:** Score **0.662** 🥉 (Rank 312 / 4,150 teams globally - Top 7.5%, Solid Bronze Medal)
> **Silver Medal Cutoff (Rank 207):** 0.663 (Distance: only 0.001 points / 1-2 correct predictions)
> **Top 10 Global Standing:** 0.704 - 0.737 (Only 11 teams worldwide are >= 0.700)

---

## 1. ESSENTIAL DOCUMENTS TO READ FIRST (CONTEXT HYPER-CORE)

If you are an AI assistant analyzing this project, please start by reading these 6 markdown documents:
1. `abc PROBLEM_STATEMENT.md` - Complete problem definition, 18 model classes, macro-recall metric, hidden test constraints, and Sinkhorn N/18 column conservation.
2. `abc LEADERBOARD.md` - Complete global leaderboard distribution (4,150 teams), the 430-team traffic jam at 0.662-0.664, medal boundaries, and Top 10 architecture secrets.
3. `abc SUBMISSIONS.md` - Chronology of all 60 submissions (33 scored submissions from V01 to V60), with architectural changes and experimental gains.
4. `abc SUBMISSIONS FAIL.md` - Comprehensive post-mortem of all 8 catastrophic failures and performance drops (9-hour GPU timeout, PCIe bottleneck, bfloat16 T4 trap, logit drift).
5. `abc DISCUSSIONS.md` - Kaggle community forum intelligence, Alex Chilton's noise floor analysis (±0.009 noise), Judith007's 0.664 formula, and Top 10 breakdown.
6. `abc CHAT_HISTORY.md` - Full engineering trajectory of 159 turns documenting every design decision, bugfix, and breakthrough.

---

## 2. TOP CHAMPION NOTEBOOKS (HIGH WATERMARKS)

- `champion_v59__champion-v59.ipynb` - Current Record Holder (**0.662** 🥉👑): Golden Anchor V50 + Prompt-Aware Relative Length Bias + Soft Injective Collision Dispersion + Sinkhorn 0.45.
- `champion_v60__champion-v60.ipynb` - High Watermark Replication (**0.662** 🥉): Apex Synergy with Soft Typography Guards + LogSumExp Sibling Sharpening.
- `champion_v58__champion-v58.ipynb` - Baseline Preserver (**0.659**): Expanded ATLAS 22% Sweet Spot + LogSumExp Sibling Conservation Gate.
- `champion_v50__champion-v50.ipynb` - Milestone Titan (**0.659**): Pure Apex Sovereign with Golden Ratio Meta Llama-3.1-8B + ModernBERT + Qwen 14B.
- `champion_v42__champion-v42.ipynb` - Historic Baseline (**0.658**): Golden Dual-Titan with Anchor Hungarian.

---

## 3. REVERSE-ENGINEERED COMPETITOR NOTEBOOKS

- `scratch__0-664-hyperparameter-tuned-who-are-u__0-664-hyperparameter-tuned-who-are-u.ipynb` - Judith007's notebook that achieved **0.664** (Rank 87) by tuning Sinkhorn strength to 0.60 and evidence clip to 1.2.
- `scratch__name-that-llm-0-662-all-public-what-failed__name-that-llm-0-662-all-public-what-failed.ipynb` - Alex Chilton's post-mortem demonstrating ±0.009 public test noise and why Hard Hungarian fails on sparse test.
- `scratch__0-663-0-660-v31-atlas-20__0-663-0-660-v31-atlas-20.ipynb` - Omer Faruk Merey's 0.663 notebook.

---

## 4. CORE REFERENCE DATASETS INCLUDED

- `data__train.csv` - Official Kaggle competition training set (2,699 text passages with labels and prompts).
- `data__test.csv` - Public test set structure (3 rows fixture for notebook commit).
- `data__prompts.csv` - Reference prompt mappings.

---

## 5. FILE NAMING CONVENTION

All nested folder paths have been flattened by replacing directory separators (`/` or `\\\\`) with double underscores (`__`).
Example: `champion_v59/champion-v59.ipynb` -> `champion_v59__champion-v59.ipynb`.
Total files included: {len(copied_files) + 1} files | Total size: {(total_bytes) / (1024**2):.2f} MB.
"""
    
    readme_path = os.path.join(dest_dir, '_AI_README_AND_INDEX.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    print('Generated _AI_README_AND_INDEX.md successfully.')

    # Create the zip archive
    print(f'Creating compressed ZIP archive at: {dest_zip}')
    with zipfile.ZipFile(dest_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(dest_dir):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, dest_dir)
                zf.write(full_path, arcname)

    zip_size = os.path.getsize(dest_zip) / (1024**2)
    print(f'ZIP Archive created successfully: {zip_size:.2f} MB')

if __name__ == '__main__':
    copy_tier1_flat()

