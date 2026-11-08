import os
import zipfile

def convert_to_txt():
    downloads_dir = os.path.expanduser('~/Downloads')
    dest_dir = os.path.join(downloads_dir, 'kaggle_ntl_AI_READY')
    dest_zip = os.path.join(downloads_dir, 'kaggle_ntl_AI_READY.zip')

    if not os.path.exists(dest_dir):
        print(f'Error: {dest_dir} does not exist!')
        return

    files = [f for f in os.listdir(dest_dir) if os.path.isfile(os.path.join(dest_dir, f))]
    print(f'Starting conversion on {len(files)} files in {dest_dir}...')

    renamed_count = 0
    for f in files:
        if not f.endswith('.txt'):
            old_path = os.path.join(dest_dir, f)
            new_name = os.path.splitext(f)[0] + '.txt'
            new_path = os.path.join(dest_dir, new_name)
            
            # If target already exists and is different, handle safely
            if old_path != new_path:
                os.rename(old_path, new_path)
                renamed_count += 1

    final_files = [f for f in os.listdir(dest_dir) if os.path.isfile(os.path.join(dest_dir, f))]
    print(f'Renamed {renamed_count} files to .txt.')
    print(f'Final file count in {dest_dir}: {len(final_files)}')

    # Verify that all files end in .txt
    non_txt = [f for f in final_files if not f.endswith('.txt')]
    print(f'Non-txt files count: {len(non_txt)}')
    assert len(non_txt) == 0, f'Found non-txt files: {non_txt}'
    assert len(final_files) == 500, f'Expected 500 files, got {len(final_files)}'

    # Update index file _AI_README_AND_INDEX.txt
    readme_path = os.path.join(dest_dir, '_AI_README_AND_INDEX.txt')
    total_bytes = sum(os.path.getsize(os.path.join(dest_dir, f)) for f in final_files)

    index_content = f"""# KAGGLE NAME THAT LLM - 500 MOST ESSENTIAL FILES (TXT FORMAT FOR AI CONSULTATION)

> **Curated & Converted to 100% .txt format for external AI consultation (ChatGPT, Claude Projects, NotebookLM, Gemini Advanced, DeepSeek R1)**
> **Total Files:** Exactly 500 files (100% pure .txt format, compatible with every AI upload interface).
> **Total Size:** {total_bytes / (1024**2):.2f} MB.
> **Evaluation Metric:** Macro-Averaged Recall (Balanced Accuracy across 18 frontier LLMs).
> **Team:** private / Mohamed Morsi (`baonguyenduyquoc`) | Team ID: 17137218.
> **Current Team Standing:** Score **0.662** 🥉 (Rank 312 / 4,150 teams globally - Top 7.5%, Solid Bronze Medal).
> **Silver Medal Cutoff (Rank 207):** **0.663** (Distance: only 0.001 points / 1-2 correct predictions!).
> **Top 10 Global Standing:** 0.704 - 0.737 (Only 11 teams worldwide are >= 0.700).

---

## 1. CORE DOCUMENTS TO READ FIRST (START HERE!)

1. `abc PROBLEM_STATEMENT.txt` - Complete problem definition, 18 model classes, macro-recall metric, hidden test constraints, and Sinkhorn N/18 column conservation.
2. `abc LEADERBOARD.txt` - Complete global leaderboard distribution (4,150 teams), the 430-team traffic jam at 0.662-0.664, medal boundaries, and Top 10 architecture secrets.
3. `abc SUBMISSIONS.txt` - Chronology of all 60 submissions (33 scored submissions from V01 to V60), with architectural changes and experimental gains.
4. `abc SUBMISSIONS FAIL.txt` - Comprehensive post-mortem of all 8 catastrophic failures and performance drops (9-hour GPU timeout, PCIe bottleneck, bfloat16 T4 trap, logit drift).
5. `abc DISCUSSIONS.txt` - Kaggle community forum intelligence, Alex Chilton's noise floor analysis (±0.009 noise), Judith007's 0.664 formula, and Top 10 breakdown.
6. `abc CHAT_HISTORY.txt` - Full engineering trajectory documenting every design decision, bugfix, and breakthrough.

---

## 2. TOP CHAMPION NOTEBOOKS (NOW IN .TXT FORMAT)

- `champion_v59__champion-v59.txt` - Current Record Holder (**0.662** 🥉👑): Golden Anchor V50 + Prompt-Aware Relative Length Bias + Soft Injective Collision Dispersion + Sinkhorn 0.45.
- `champion_v60__champion-v60.txt` - High Watermark Replication (**0.662** 🥉): Apex Synergy with Soft Typography Guards + LogSumExp Sibling Sharpening.
- `champion_v58__champion-v58.txt` - Baseline Preserver (**0.659**): Expanded ATLAS 22% Sweet Spot + LogSumExp Sibling Conservation Gate.
- `champion_v50__champion-v50.txt` - Milestone Titan (**0.659**): Pure Apex Sovereign with Golden Ratio Meta Llama-3.1-8B + ModernBERT + Qwen 14B.
- `champion_v42__champion-v42.txt` - Historic Baseline (**0.658**): Golden Dual-Titan with Anchor Hungarian.

---

## 3. REVERSE-ENGINEERED COMPETITOR NOTEBOOKS (.TXT FORMAT)

- `scratch__0-664-hyperparameter-tuned-who-are-u__0-664-hyperparameter-tuned-who-are-u.txt` - Judith007's notebook that achieved **0.664** (Rank 87) by tuning Sinkhorn strength to 0.60 and evidence clip to 1.2.
- `scratch__name-that-llm-0-662-all-public-what-failed__name-that-llm-0-662-all-public-what-failed.txt` - Alex Chilton's post-mortem demonstrating ±0.009 public test noise and why Hard Hungarian fails on sparse test.
- `scratch__0-663-0-660-v31-atlas-20__0-663-0-660-v31-atlas-20.txt` - Omer Faruk Merey's 0.663 notebook.
- `scratch__0-662-qwen3-8-27b-lora-weighted-ensemble__0-662-qwen3-8-27b-lora-weighted-ensemble.txt` - Vladislav Lavrentev's 0.662 notebook.

---

## 4. CORE REFERENCE DATASETS (.TXT FORMAT)

- `data__train.txt` - Official Kaggle competition training set (2,699 text passages with labels and prompts in CSV format).
- `data__test.txt` - Public test set structure (3 rows fixture for notebook commit).
- `data__prompts.txt` - Reference prompt mappings.

---

## 5. FILE NAMING CONVENTION

All nested folder paths have been flattened by replacing directory separators (`/` or `\\\\`) with double underscores (`__`), and all file extensions have been converted to pure `.txt` format.
"""
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    print('Updated _AI_README_AND_INDEX.txt.')

    # Recreate the zip archive with all 500 .txt files
    print(f'Recreating compressed ZIP archive at: {dest_zip}...')
    if os.path.exists(dest_zip):
        os.remove(dest_zip)
        
    with zipfile.ZipFile(dest_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for file in final_files:
            full_path = os.path.join(dest_dir, file)
            zf.write(full_path, file)

    zip_size = os.path.getsize(dest_zip) / (1024**2)
    print(f'ZIP Archive updated successfully: {zip_size:.2f} MB with exactly {len(final_files)} .txt files.')

if __name__ == '__main__':
    convert_to_txt()

