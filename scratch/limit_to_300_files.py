import os
import zipfile

def limit_to_300():
    downloads_dir = os.path.expanduser('~/Downloads')
    dest_dir = os.path.join(downloads_dir, 'kaggle_ntl_AI_READY')
    dest_zip = os.path.join(downloads_dir, 'kaggle_ntl_AI_READY.zip')

    if not os.path.exists(dest_dir):
        print(f'Error: {dest_dir} does not exist!')
        return

    files = [f for f in os.listdir(dest_dir) if os.path.isfile(os.path.join(dest_dir, f))]
    print(f'Initial file count in {dest_dir}: {len(files)}')

    def score_300(f):
        # 1. Master AI Index
        if f.startswith('_AI'): return 1000
        # 2. Strategic Markdown
        if f.startswith('abc'): return 950
        # 3. Ground truth competition data
        if f.startswith('data__'): return 900
        # 4. All competitor and community intelligence notebooks
        competitor_keywords = [
            '0-664', '0-663', '0-662', 'chilton', 'judith', 'merey', 'lavrentev',
            'prvsiyan', 'areebbb', 'lightning', 'nishant', 'fleongg', 'adamski',
            'kirilltym', 'drozanes', 'kozykappa', 'romantamrazov', 'maverickss26',
            'dhruv', 'gopain', 'who-am-i', 'fully-reproducible', 'teach-the-model'
        ]
        if f.startswith('scratch') and any(k in f.lower() for k in competitor_keywords):
            if 'kernel-metadata' not in f:
                return 850

        # 5. Champion notebooks by version tiers (excluding kernel-metadata)
        if f.startswith('champion_') and 'kernel-metadata' not in f:
            v_num = 0
            try:
                v_str = f.split('__')[0].replace('champion_v', '')
                v_num = int(v_str)
            except: pass
            if v_num >= 40: return 800 + v_num   # V40-V60 (The Modern SOTA Era)
            elif v_num >= 20: return 750 + v_num # V20-V39 (The Silver Foundation)
            else: return 700 + v_num             # V01-V19 (Early Baselines)
            
        # 6. Key algorithmic test and verification modules
        algo_keywords = [
            'hungarian', 'anchor', 'sibling', 'sinkhorn', 'atlas', 'typography',
            'stylometry', 'length', 'cluster', 'eda', 'bayes', 'likelihood', 'dispersion'
        ]
        if any(k in f.lower() for k in algo_keywords):
            return 740

        # 7. Modern version build scripts (v25 to v60)
        if 'build_' in f and any(f'_{k}' in f for k in range(25, 65)):
            return 710

        # 8. Downloaded dataset schemas and summary statistics
        if f.startswith('downloaded_datasets') and any(k in f for k in ['summary', 'stats', 'sources', 'style_gap', 'train_vs', 'label_summary']):
            return 650

        # 9. Low value: kernel-metadata.txt files (just 5 lines of json)
        if 'kernel-metadata' in f:
            return 20

        # 10. Low value: polling, temporary checks, print scripts
        low_patterns = ['auto_submit', 'beep', 'wait', 'lint', 'check_', 'find_', 'dump_', 'track_', 'fix_']
        if any(k in f for k in low_patterns):
            return 10

        return 400

    scored = [(f, score_300(f)) for f in files]
    scored.sort(key=lambda x: (x[1], x[0]), reverse=True)

    TARGET_COUNT = 300
    keep_list = set(f for f, score in scored[:TARGET_COUNT])
    remove_list = [f for f, score in scored[TARGET_COUNT:]]

    print(f'Keeping: {len(keep_list)} files | Removing: {len(remove_list)} files')

    # Remove the 200 files
    removed_count = 0
    for f in remove_list:
        fp = os.path.join(dest_dir, f)
        try:
            os.remove(fp)
            removed_count += 1
        except Exception as e:
            print(f'Error removing {f}: {e}')

    final_files = [f for f in os.listdir(dest_dir) if os.path.isfile(os.path.join(dest_dir, f))]
    print(f'Verified remaining files in {dest_dir}: {len(final_files)}')
    assert len(final_files) == 300, f'Expected 300 files, got {len(final_files)}'

    total_bytes = sum(os.path.getsize(os.path.join(dest_dir, f)) for f in final_files)
    print(f'Total size of 300 files: {total_bytes / (1024**2):.2f} MB')

    # Update index file _AI_README_AND_INDEX.txt
    readme_path = os.path.join(dest_dir, '_AI_README_AND_INDEX.txt')
    index_content = f"""# KAGGLE NAME THAT LLM - 300 MOST ESSENTIAL FILES (HYPER-DENSE AI CONSULTATION PACK)

> **Curated, Prioritized & Converted to 100% .txt format for external AI consultation (ChatGPT, Claude Projects, NotebookLM, Gemini Advanced, DeepSeek R1)**
> **Total Files:** Exactly 300 most critical files (all metadata and temporary scripts pruned).
> **Total Size:** {total_bytes / (1024**2):.2f} MB (High-density signal pack).
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

## 2. TOP CHAMPION NOTEBOOKS (HIGH WATERMARKS)

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
- `scratch__romantamrazov__name-that-llm-who-am-i.txt` - Roman Tamrazov's 0.654 baseline.
- `scratch__dhruv_repro__name-that-llm-fully-reproducible.txt` - Dhruv Tibarewal's 0.651 baseline.
- `scratch__gopain_grouping__when-does-test-time-prompt-grouping-help.txt` - GoPain's prompt grouping analysis.

---

## 4. CORE REFERENCE DATASETS (.TXT FORMAT)

- `data__train.txt` - Official Kaggle competition training set (2,699 text passages with labels and prompts in CSV format).
- `data__test.txt` - Public test set structure (3 rows fixture for notebook commit).
- `data__prompts.txt` - Reference prompt mappings.

---

## 5. FILE NAMING CONVENTION

All nested folder paths have been flattened by replacing directory separators (`/` or `\\\\`) with double underscores (`__`), and all file extensions are pure `.txt` format.
Total files in this pack: Exactly 300 files.
"""
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    print('Updated _AI_README_AND_INDEX.txt.')

    # Recreate the zip archive with exact 300 files
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
    limit_to_300()

