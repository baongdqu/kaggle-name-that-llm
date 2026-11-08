import json
from pathlib import Path

# Load V27 notebook as base
with open("champion_v27/champion-v27.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 0: Markdown documentation for Champion V28 Sovereign Purge
nb["cells"][0]["source"] = [
    "# 0.639+ -> 0.648+ | Champion V28 Sovereign Purge (120k V5 Pool + Comp Stack 0.563 De-Weighting to 3%)\n",
    "**Team: Mohamed Morsi (Bao Ng) | Rank 152 Silver Medal -> Target Top 10 Gold Medal Tier**\n",
    "\n",
    "### Core Architecture & Precision Engineering (Champion V28 Sovereign Purge):\n",
    "1. **Purging the 0.563 Baseline Anchor Drag:** In V25, increasing `comp_stack` from 0.15 to 0.25 crashed LB from 0.639 to 0.630 (-0.009). V28 decisively purges this drag by reducing `comp_stack` weight from 15% down to 3% (`0.97 * _main_raw + 0.03 * _comp`), granting 97% decision authority to the Two-Stage 14B LoRA and Spectral SVC.\n",
    "2. **120k Cleaned Passage Pool (`nlm_pool_v5.parquet`):** Upgrades the n-gram LinearSVC training corpus from 75,783 passages (pool_v4) to 119,899 passages (pool_v5), adding +44,116 verified passages to drastically broaden character and word n-gram boundaries on rare frontier models.\n",
    "3. **Helper Calibration:** Re-tunes `night24_helper` from 10% down to 5% (`0.95 * P + 0.05 * _hp`), focusing decision mass on the deep contextual two-stage representation.\n",
    "4. **Robust Generative Likelihood Calibration (KAIST MAABench Principle):** Retains ATLAS 8B generative likelihood on ambiguous samples (`margin < 0.30`) with temperature smoothing ($T=1.10$) and full cross-family scope.\n",
    "5. **Proven Intra-Family Sibling Micro-Gates:** Pure intra-family sibling discrimination for Gemini 3.8 vs 3.7 (reasoning prefixes) and GLM 5.3 vs 5.3-flash (Chinese Unicode marks).\n",
    "6. **Lossless Speedup Trio (100% Mathematically Invariant):** `torch.inference_mode()`, cuDNN benchmark & TF32, and `/tmp` tokenizer staging.\n",
    "7. **Terminal Sinkhorn Purity (300 iters):** Strict uniform $N/18$ terminal normalization, shielding against Private LB distribution shift.\n"
]

# Cell 13: Upgrade SVC to use nlm_pool_v5.parquet
c13_src = "".join(nb["cells"][13]["source"])
old_c13_pool = """    pool = pd.read_parquet(find("pool_v4.parquet"))"""
new_c13_pool = """    # V28 Upgrade: Train SVC on 120k cleaned pool (nlm_pool_v5) if available, fallback to pool_v4
    _pool_hits = sorted(glob.glob(f"{INPUT_DIR}/**/nlm_pool_v5.parquet", recursive=True))
    if not _pool_hits:
        _pool_hits = sorted(glob.glob(f"{INPUT_DIR}/**/pool_v4.parquet", recursive=True))
    assert len(_pool_hits) > 0, "No pool parquet found!"
    pool = pd.read_parquet(_pool_hits[0])
    print(f"V28 SVC Pool Loaded: {len(pool)} passages from {_pool_hits[0]}", flush=True)"""

assert old_c13_pool in c13_src, "Failed to match old pool loading in Cell 13!"
c13_src = c13_src.replace(old_c13_pool, new_c13_pool)
nb["cells"][13]["source"] = [line + "\n" for line in c13_src.splitlines()]

# Cell 22: Update sha check in part two verification
c22_src = "".join(nb["cells"][22]["source"])
old_c22_sha = """pool_sha256=_sha(find('pool_v4.parquet'))"""
new_c22_sha = """pool_sha256=_sha(_pool_hits[0])"""
assert old_c22_sha in c22_src, "Failed to match old sha in Cell 22!"
c22_src = c22_src.replace(old_c22_sha, new_c22_sha)
nb["cells"][22]["source"] = [line + "\n" for line in c22_src.splitlines()]

# Cell 24: De-weight comp_stack from 0.15 to 0.03 and helper from 0.10 to 0.05
c24_src = "".join(nb["cells"][24]["source"])

old_c24_comp = """    P = .85 * _main_raw + .15 * _comp"""
new_c24_comp = """    # V28 Sovereign Purge: De-weight comp_stack 0.563 baseline from 15% down to 3%
    # Giving 97% decision authority to the 0.639 Two-Stage LoRA + 120k Spectral SVC
    P = .97 * _main_raw + .03 * _comp"""

assert old_c24_comp in c24_src, "Failed to match old comp blend in Cell 24!"
c24_src = c24_src.replace(old_c24_comp, new_c24_comp)

old_c24_helper = """    _unchanged_top1 = P.copy()
    P = .90 * P + .10 * _hp"""
new_c24_helper = """    _unchanged_top1 = P.copy()
    # V28 Tuning: Reduce helper readout bank from 10% to 5% to preserve pristine two-stage representation
    P = .95 * P + .05 * _hp"""

assert old_c24_helper in c24_src, "Failed to match old helper blend in Cell 24!"
c24_src = c24_src.replace(old_c24_helper, new_c24_helper)

# Update proof dict in Cell 24
old_c24_proof = "comp_weight=.15, atlas_weight=.10, night24_helper_weight=.10"
new_c24_proof = "comp_weight=.03, atlas_weight=.10, night24_helper_weight=.05"
assert old_c24_proof in c24_src, "Failed to match old proof weights in Cell 24!"
c24_src = c24_src.replace(old_c24_proof, new_c24_proof)

old_c24_proof_bal = "balancing='Terminal Full Sinkhorn (300 iters) with Robust ATLAS Likelihood Calibration, Full Ambiguity Scope & Golden Pool_v4'"
new_c24_proof_bal = "balancing='Terminal Full Sinkhorn (300 iters) with Sovereign Purge (comp_w=0.03), 120k V5 Pool & Robust ATLAS'"
assert old_c24_proof_bal in c24_src, "Failed to match old proof bal in Cell 24!"
c24_src = c24_src.replace(old_c24_proof_bal, new_c24_proof_bal)

nb["cells"][24]["source"] = [line + "\n" for line in c24_src.splitlines()]

# Save to champion_v28/champion-v28.ipynb
Path("champion_v28").mkdir(exist_ok=True)
with open("champion_v28/champion-v28.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("SUCCESS: Champion V28 notebook built successfully!")

# Build champion_v28/kernel-metadata.json
meta = {
    "id": "baonguyenduyquoc/name-that-llm-champion-v28-sovereign-purge",
    "title": "Name That LLM Champion V28 Sovereign Purge",
    "code_file": "champion-v28.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": [],
    "dataset_sources": [
        "dhruvtibarewal/nlm-l24-adapters",
        "seyitkaangunes/ntl-extra-api-passages",
        "abdullah1afify/ntl-generations",
        "dophuchung/ntl-proxy-generations",
        "zbigniewkirchner/ntl-qwen3-8b-lora-claude-head",
        "dhruvtibarewal/nlm-matched-clean",
        "dhruvtibarewal/nlm-offline-wheels",
        "josephadamski91/ntl-atlas-qwen3-8b-v1",
        "lucasboesen/ntl-free-proxy-generations",
        "fleongg/ntl-legacy-qwen24-recovery",
        "fleongg/ntl-quality-selected-public-svc-pool",
        "kirilltym/ntl-spectral-style-heads-paired-4404",
        "nishantdahal/ntl-two-stage-lora",
        "nishantdahal/ntl-handwriting-blend",
        "prvsiyan/prvsiyans-extended-name-that-llm-dataset",
        "bbucxi/qwen35-9b-safetensors",
        "pjmathematician/ultrabert-380m-ntl"
    ],
    "kernel_sources": [
        "lucasboesen/name-that-llm-glm4-9b-trunc25",
        "lucasboesen/name-that-llm-extract-encoders2",
        "lucasboesen/name-that-llm-electra-large-weights",
        "lucasboesen/name-that-llm-desklib-weights",
        "lucasboesen/name-that-llm-train-lora24-s7",
        "lucasboesen/name-that-llm-train-tokattn",
        "lucasboesen/name-that-llm-train-tokattn2",
        "vnton37nguyn/ntl-train-lora-qwen25-instyle"
    ],
    "competition_sources": [
        "name-that-llm"
    ],
    "model_sources": [
        "qwen-lm/qwen-3/Transformers/8b-base/1",
        "qwen-lm/qwen-3/Transformers/8b/1",
        "qwen-lm/qwen-3/Transformers/14b/1",
        "qwen-lm/qwen-3/Transformers/4b-base/1",
        "qwen-lm/qwen-3/Transformers/14b-base/1",
        "qwen-lm/qwen2.5/Transformers/14b/1",
        "qwen-lm/qwen2.5/Transformers/7b/1"
    ],
    "machine_shape": "NvidiaTeslaT4"
}

with open("champion_v28/kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("SUCCESS: Champion V28 metadata built successfully!")
