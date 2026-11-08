import json
import os

def build_v61():
    src_nb_path = 'champion_v59/champion-v59.ipynb'
    out_dir = 'champion_v61'
    os.makedirs(out_dir, exist_ok=True)
    out_nb_path = os.path.join(out_dir, 'champion-v61.ipynb')

    with open(src_nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Update Cell 0: Title & Strategy
    nb['cells'][0]['source'] = [
        "# Name That LLM Champion V61 Judith-Sovereign Apex (0.664+ SOTA Breakout)\n",
        "# Global Silver Medal Breakthrough (Score 0.664+ Target)\n",
        "#\n",
        "# Architecture Highlights:\n",
        "# 1. V59 High Watermark Core (0.662 Record): Prompt-Aware Relative Length Bias + Soft Injective Dispersion\n",
        "# 2. Judith007 SOTA Tuning (Rank 87): Marginal Sinkhorn 0.60 + Evidence Clip 1.2 (target_mix 0.35)\n",
        "# 3. Boosted Qwen 2.5 14B (w=1.0 -> 2.0) - Strongest Open Causal Backbone\n",
        "# 4. Golden Dual-Titan Backbone: Qwen 14B (w=3.0) + Llama 3.1 8B (w=1.8 dp:True) + ModernBERT-Large (w=2.0)\n",
        "# 5. Strict Anti-Hungarian Discipline (--perpassage 1)\n",
        "\n",
        "print('Champion V61 Judith-Sovereign Apex initialized.')\n"
    ]

    # 2. Update Cell 2 Header
    cell2_src = ''.join(nb['cells'][2]['source'])
    cell2_src = cell2_src.replace('Champion V59 Prompt-Aware Relative Sovereign', 'Champion V61 Judith-Sovereign Apex')
    nb['cells'][2]['source'] = [cell2_src]

    # 3. Update Cell 6 (infer.py): Evidence Clip 1.5 -> 1.2 (Judith007 formula)
    cell6_src = ''.join(nb['cells'][6]['source'])
    old_clip = "_evidence = np.clip(1.5 * _evidence / max(float(_fw.sum()), 1e-6), -0.30, 0.30)"
    new_clip = "_evidence = np.clip(1.2 * _evidence / max(float(_fw.sum()), 1e-6), -0.30, 0.30)"
    assert old_clip in cell6_src, 'Target clip string not found in Cell 6!'
    cell6_src = cell6_src.replace(old_clip, new_clip)
    nb['cells'][6]['source'] = [cell6_src]

    # 4. Update Cell 12: Sinkhorn 0.45 -> 0.60, Qwen 2.5 14B weight 1.0 -> 2.0
    cell12_src = ''.join(nb['cells'][12]['source'])
    
    # Update sinkhorn argument
    old_sk = "'--adapt_thr', '0.15', '--sinkhorn', '0.45', '--sinkhorn_target_mix', '0.35'"
    new_sk = "'--adapt_thr', '0.15', '--sinkhorn', '0.60', '--sinkhorn_target_mix', '0.35'"
    assert old_sk in cell12_src, 'Target sinkhorn arg string not found in Cell 12!'
    cell12_src = cell12_src.replace(old_sk, new_sk)

    # Boost Qwen 2.5 14B weight
    old_q25 = '{"name": "q25_14", "type": "l24", "path": B25_14, "adapters": [Q25_14], "keep": Q25_14C[\'keep\'], "layers": Q25_14C[\'layers\'], "w": 1.0,'
    new_q25 = '{"name": "q25_14", "type": "l24", "path": B25_14, "adapters": [Q25_14], "keep": Q25_14C[\'keep\'], "layers": Q25_14C[\'layers\'], "w": 2.0,'
    assert old_q25 in cell12_src, 'Target Qwen 2.5 14B definition not found in Cell 12!'
    cell12_src = cell12_src.replace(old_q25, new_q25)

    nb['cells'][12]['source'] = [cell12_src]

    # Write notebook
    with open(out_nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print(f'Successfully built {out_nb_path} from V59!')

    # Update metadata
    meta_path = os.path.join(out_dir, 'kernel-metadata.json')
    with open('champion_v59/kernel-metadata.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    meta['id'] = 'baonguyenduyquoc/ntl-champion-v61-judith-sovereign-apex'
    meta['title'] = 'NTL Champion V61 Judith Sovereign Apex'
    meta['code_file'] = 'champion-v61.ipynb'
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'Successfully updated {meta_path}. Title length: {len(meta["title"])}')

if __name__ == '__main__':
    build_v61()

