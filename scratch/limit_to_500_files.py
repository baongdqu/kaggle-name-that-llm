import os
import shutil
import zipfile

def limit_to_500():
    downloads_dir = os.path.expanduser('~/Downloads')
    dest_dir = os.path.join(downloads_dir, 'kaggle_ntl_AI_READY')
    dest_zip = os.path.join(downloads_dir, 'kaggle_ntl_AI_READY.zip')

    if not os.path.exists(dest_dir):
        print(f'Error: {dest_dir} does not exist!')
        return

    files = [f for f in os.listdir(dest_dir) if os.path.isfile(os.path.join(dest_dir, f))]
    print(f'Initial file count in {dest_dir}: {len(files)}')

    def file_score(f):
        # 1. Master Readme & Index
        if f.startswith('_AI'): return 100
        # 2. Strategic Markdown documents
        if f.startswith('abc'): return 95
        # 3. Competition Data (train, test, prompts)
        if f.startswith('data__'): return 90
        # 4. All Champion Notebooks & Configs (all 60 versions)
        if f.startswith('champion_'): return 85
        # 5. Reverse-engineered competitor notebooks in scratch (0-664, 0-663, 0-662, etc.)
        if f.startswith('scratch') and f.endswith('.ipynb'): return 80
        # 6. Modern version build scripts (v25 to v60)
        if 'build_' in f and any(f'_{k}' in f for k in range(25, 65)): return 75
        # 7. Key algorithmic test & verification modules
        if any(k in f for k in ['hungarian', 'anchor', 'sibling', 'sinkhorn', 'atlas', 'modernbert', 'typography', 'stylometry', 'length', 'cluster']):
            return 70
        # 8. Downloaded dataset schemas & summary statistics
        if f.startswith('downloaded_datasets'): return 65
        
        # 9. Scratch python scripts - separate real utility from throwaway scripts
        if f.startswith('scratch') and f.endswith('.py'):
            low_patterns = [
                'append_turn', 'append_history', 'append_v',
                'monitor_', 'beep_', 'wait_',
                'lint_', 'verify_', '_syntax',
                'update_sub', 'update_score', 'update_leaderboard',
                'update_chat', 'update_roadmap', 'update_top', 'update_v',
                'inspect_', 'check_', 'print_', 'read_',
                'push_', 'make_', 'patch_', 'find_', 'dump_', 'track_',
                'diff_', 'extract_', 'search_', 'compare_'
            ]
            if any(p in f for p in low_patterns):
                return 10
            return 50

        # Low priority redundant leaderboard CSV dumps
        if f.endswith('.csv') and 'leaderboard' in f.lower():
            return 15

        return 40

    # Score all files
    scored = [(f, file_score(f)) for f in files]
    # Sort descending by score, then alphabetically
    scored.sort(key=lambda x: (x[1], x[0]), reverse=True)

    TARGET_COUNT = 500
    keep_list = set(f for f, score in scored[:TARGET_COUNT])
    remove_list = [f for f, score in scored[TARGET_COUNT:]]

    print(f'Keeping: {len(keep_list)} files | Removing: {len(remove_list)} files')

    # Remove the low-value files
    removed_count = 0
    for f in remove_list:
        fp = os.path.join(dest_dir, f)
        try:
            os.remove(fp)
            removed_count += 1
        except Exception as e:
            print(f'Error removing {f}: {e}')

    remaining_files = [f for f in os.listdir(dest_dir) if os.path.isfile(os.path.join(dest_dir, f))]
    print(f'Verified remaining files in {dest_dir}: {len(remaining_files)}')

    total_bytes = sum(os.path.getsize(os.path.join(dest_dir, f)) for f in remaining_files)
    print(f'Total size of 500 files: {total_bytes / (1024**2):.2f} MB')

    # Update _AI_README_AND_INDEX.md with exact 500 files notice
    readme_path = os.path.join(dest_dir, '_AI_README_AND_INDEX.md')
    index_content = f"""# KAGGLE NAME THAT LLM - 500 MOST ESSENTIAL FILES (AI CONSULTATION PACK)

> **Curated & Optimized for external AI consultation (ChatGPT, Claude Projects, Gemini Advanced, DeepSeek R1)**
> **Total Files:** Exactly {len(remaining_files)} most critical files (all low-value throwaway scripts pruned).
> **Total Size:** {total_bytes / (1024**2):.2f} MB (Extremely compact, fits into any AI context/upload limit).
> **Evaluation Metric:** Macro-Averaged Recall (Balanced Accuracy across 18 frontier LLMs).
> **Team:** private / Mohamed Morsi (`baonguyenduyquoc`) | Team ID: 17137218.
> **Current Team Standing:** Score **0.662** 🥉 (Rank 312 / 4,150 teams globally - Top 7.5%, Solid Bronze Medal).
> **Silver Medal Cutoff (Rank 207):** **0.663** (Distance: only 0.001 points / 1-2 correct predictions!).
> **Top 10 Global Standing:** 0.704 - 0.737 (Only 11 teams worldwide are >= 0.700).

---

## 1. CORE DOCUMENTS TO READ FIRST (START HERE!)

1. `abc PROBLEM_STATEMENT.md` - Complete problem definition, 18 model classes, macro-recall metric, hidden test constraints, and Sinkhorn N/18 column conservation.
2. `abc LEADERBOARD.md` - Complete global leaderboard distribution (4,150 teams), the 430-team traffic jam at 0.662-0.664, medal boundaries, and Top 10 architecture secrets.
3. `abc SUBMISSIONS.md` - Chronology of all 60 submissions (33 scored submissions from V01 to V60), with architectural changes and experimental gains.
4. `abc SUBMISSIONS FAIL.md` - Comprehensive post-mortem of all 8 catastrophic failures and performance drops (9-hour GPU timeout, PCIe bottleneck, bfloat16 T4 trap, logit drift).
5. `abc DISCUSSIONS.md` - Kaggle community forum intelligence, Alex Chilton's noise floor analysis (±0.009 noise), Judith007's 0.664 formula, and Top 10 breakdown.
6. `abc CHAT_HISTORY.md` - Full engineering trajectory documenting every design decision, bugfix, and breakthrough.

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
- `scratch__0-662-qwen3-8-27b-lora-weighted-ensemble__0-662-qwen3-8-27b-lora-weighted-ensemble.ipynb` - Vladislav Lavrentev's 0.662 notebook.

---

## 4. CORE REFERENCE DATASETS INCLUDED

- `data__train.csv` - Official Kaggle competition training set (2,699 text passages with labels and prompts).
- `data__test.csv` - Public test set structure (3 rows fixture for notebook commit).
- `data__prompts.csv` - Reference prompt mappings.

---

## 5. FILE NAMING CONVENTION

All nested folder paths have been flattened by replacing directory separators (`/` or `\\\\`) with double underscores (`__`).
Example: `champion_v59/champion-v59.ipynb` -> `champion_v59__champion-v59.ipynb`.
"""
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    print('Updated _AI_README_AND_INDEX.md.')

    # Recreate the zip archive with exact 500 files
    print(f'Recreating compressed ZIP archive at: {dest_zip}')
    with zipfile.ZipFile(dest_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for file in remaining_files:
            full_path = os.path.join(dest_dir, file)
            zf.write(full_path, file)

    zip_size = os.path.getsize(dest_zip) / (1024**2)
    print(f'ZIP Archive updated successfully: {zip_size:.2f} MB with exactly {len(remaining_files)} files.')

if __name__ == '__main__':
    limit_to_500()

