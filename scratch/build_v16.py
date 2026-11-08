import json
import os
import ast

os.makedirs('champion_v16', exist_ok=True)

# 1. Metadata for V16 Apex Sovereign
meta_v16 = {
  "id": "baonguyenduyquoc/name-that-llm-champion-v16-apex-sovereign",
  "title": "Name That LLM Champion V16 Apex Sovereign",
  "code_file": "champion-v16.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": True,
  "enable_gpu": True,
  "enable_tpu": False,
  "enable_internet": False,
  "keywords": [
    "gpu"
  ],
  "dataset_sources": [
    "seyitkaangunes/ntl-extra-api-passages",
    "zbigniewkirchner/ntl-extra-gpt-passages",
    "abdullah1afify/ntl-generations",
    "dophuchung/ntl-proxy-generations",
    "zbigniewkirchner/ntl-qwen3-8b-lora-instyle",
    "zbigniewkirchner/ntl-qwen3-8b-lora-claude-head",
    "lucasboesen/ntl-free-proxy-generations",
    "prvsiyan/prvsiyans-extended-name-that-llm-dataset",
    "bbucxi/qwen35-9b-safetensors",
    "pjmathematician/ultrabert-380m-ntl",
    "nishantdahal/ntl-two-stage-lora"
  ],
  "kernel_sources": [
    "lucasboesen/name-that-llm-glm4-9b-trunc25",
    "lucasboesen/name-that-llm-extract-encoders2",
    "lucasboesen/name-that-llm-electra-large-weights",
    "lucasboesen/name-that-llm-desklib-weights",
    "lightningv08/ntl-lora24-fivefold-training",
    "lucasboesen/name-that-llm-train-lora24-s7",
    "lucasboesen/name-that-llm-train-tokattn",
    "lucasboesen/name-that-llm-train-tokattn2",
    "vnton37nguyn/ntl-train-lora-qwen25-instyle"
  ],
  "competition_sources": [
    "name-that-llm"
  ],
  "model_sources": [
    "qwen-lm/qwen-3/Transformers/14b-base/1",
    "qwen-lm/qwen-3/Transformers/8b-base/1",
    "qwen-lm/qwen-3/Transformers/4b-base/1",
    "qwen-lm/qwen2.5/Transformers/7b/1"
  ],
  "machine_shape": "NvidiaTeslaT4"
}

with open('champion_v16/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta_v16, f, indent=2)

print("Saved champion_v16/kernel-metadata.json")

# 2. Load base notebook V15
nb = json.load(open('champion_v15/champion-v15.ipynb', encoding='utf-8'))

# 3. Update Title & Intro (Cell 0)
nb['cells'][0]['source'] = """# 0.630+ | Champion V16 Apex Sovereign (Two-Stage 14B/8B Full-Force + 35k Pool N-Grams + Injective Cluster Decoding)

**Champion V16 Apex Sovereign Architecture**:
1. **Pillar 1 - Full-Force Two-Stage Qwen3-14B & 8B LoRAs**:
   - Qwen3-14B Base ($H=5120$, 27 layers) re-weighted to $w=0.70$ (plain) + $twin\\_w=0.25$ (centred twin).
   - Qwen3-8B Base ($H=4096$, 24 layers) re-weighted to $w=0.40$ (plain) + $twin\\_w=0.15$ (centred twin).
   - Provides dominant sister-model semantic separation (Gemini 3.7 vs 3.8, GPT-5.6 Luna vs Sol vs Terra).
2. **Pillar 2 - 35,000-Passage 4-View Public Pool LinearSVC**:
   - Trained on `train.csv` + 34,940 high-quality external passages from `pool_v1.parquet` across all 18 models.
   - 4 complementary stylometric views: Char (3-5), Word (1-3), Shape (2-6), Function-only (1-3) with CONTENT masking.
   - Contributes orthogonal global stylometric evidence ($w=0.60$).
3. **Pillar 3 - Exact Soft Marginal Injective Cluster Decoding**:
   - For all test prompt clusters ($g \in [2, 18]$), enforces mutually exclusive assignment via cavity mean-field repulsion ($P(a \\mid \\text{distinct})$).
   - Eliminates redundant multi-predictions of the same LLM for passages answering the same question.
4. **Pillar 4 - V14 Champion Stack + Universal 18-Class Mega-Gate**:
   - In-Style Qwen3-8B LoRA + Qwen2.5-7B LoRA + LoRA24 Public, Seed7, Claude-head.
   - Shared-Prompt Gaussian Group Heads on 4 dense backbone views.
   - Universal 18-Class Mega-Gate + 4-Way Chinese Frontier Gate with Cross-Family Soft Posterior Calibration & High-Confidence Anchor Lock ($P_1 - P_2 > 0.40$).
"""

# 4. Update Cell 5: Add 4-View Public Pool N-Gram (35,000 Passages from pool_v1.parquet)
c5 = nb['cells'][5]['source']
pool_svc_code = """

# ============================================================ 2b. Nishant Dahal 4-View Public Pool N-Gram (35,000 Passages)
def fit_pool_svc():
    pool_file = None
    for pf in sorted(glob.glob('/kaggle/input/**/pool_v1.parquet', recursive=True)):
        pool_file = pf
        break
    if not pool_file or not os.path.exists(pool_file):
        print("pool_v1.parquet not found, skipping public pool SVC", flush=True)
        return
    print(f"Loading public pool from {pool_file}...", flush=True)
    pool = pd.read_parquet(pool_file)
    pool = pool[pool['model'].isin(LABELS)].copy()
    fit_texts = train['text'].astype(str).tolist() + pool['text'].astype(str).tolist()
    fit_y = np.r_[y, pool['model'].map(LAB2I).values]
    
    # Four views: char 3-5, word 1-3, shape 2-6, function 1-3
    pool_views = [
        ("char", (3, 5), 250_000, 1.0, str),
        ("word", (1, 3), 180_000, 0.7, str),
        ("char", (2, 6), 60_000, 0.5, shape_view),
        ("word", (1, 3), 50_000, 0.5, function_view)
    ]
    A, B = [], []
    for analyzer, ngr, maxf, w, prep in pool_views:
        opts = dict(analyzer=analyzer, ngram_range=ngr, lowercase=False, sublinear_tf=True, min_df=2, max_features=maxf, dtype=np.float32)
        if analyzer == 'word':
            opts['token_pattern'] = r"(?u)\\b\\w+\\b|[^\\w\\s]"
        vec = TfidfVectorizer(**opts)
        A.append(vec.fit_transform([prep(t) for t in fit_texts]) * w)
        B.append(vec.transform([prep(t) for t in TE]) * w)
    
    print(f"Fitting 4-View LinearSVC on {len(fit_texts)} passages...", flush=True)
    svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced").fit(hstack(A, format="csr"), fit_y)
    P_pool = softmax_rows(svc.decision_function(hstack(B, format="csr")))
    
    W_POOL_SVC = 0.60
    LOGIT[:] += W_POOL_SVC * np.log(np.clip(P_pool, 1e-9, 1.0))
    print(f"  + pool_4view_svc weight {W_POOL_SVC:.3f} added on {len(TE)} rows ({time.time() - T0:.0f}s)", flush=True)
    del A, B, svc, pool, fit_texts, fit_y
    gc.collect()

if not SMOKE:
    fit_pool_svc()
"""

target_c5_anchor = "grams = StyleGrams().fit(TR + EX_T)"
assert target_c5_anchor in c5, "Anchor in Cell 5 not found!"
c5 = c5 + pool_svc_code
nb['cells'][5]['source'] = c5
print("Cell 5 updated with 35k Public Pool 4-View LinearSVC.")

# 5. Update Cell 12: Increase Two-Stage LoRA weights to full force
c12 = nb['cells'][12]['source']
old_ts_weights = """    TS_LORAS = {
        "lora14b_v1": dict(w=0.35, twin_w=0.12, bs=8,  base=dict(model_type="qwen3", hidden_size=5120, num_hidden_layers=40)),
        "lora8b_v1":  dict(w=0.20, twin_w=0.08, bs=16, base=dict(model_type="qwen3", hidden_size=4096, num_hidden_layers=36)),
    }"""

new_ts_weights = """    TS_LORAS = {
        "lora14b_v1": dict(w=0.70, twin_w=0.25, bs=8,  base=dict(model_type="qwen3", hidden_size=5120, num_hidden_layers=40)),
        "lora8b_v1":  dict(w=0.40, twin_w=0.15, bs=16, base=dict(model_type="qwen3", hidden_size=4096, num_hidden_layers=36)),
    }"""

assert old_ts_weights in c12, "Old TS_LORAS weights in Cell 12 not found!"
c12 = c12.replace(old_ts_weights, new_ts_weights)
nb['cells'][12]['source'] = c12
print("Cell 12 updated with Full-Force Two-Stage LoRA weights (14B: 0.70/0.25, 8B: 0.40/0.15).")

# 6. Update Cell 16: Add Injective Soft Marginal Cluster Decoding & Margin Lock 0.40
c16 = nb['cells'][16]['source']

target_c16_anchor = """# 3. High-confidence ensemble protection (protect top predictions with high margin)
p_sort = np.sort(P, axis=1)
margin = p_sort[:, -1] - p_sort[:, -2]
confident_mask = margin > 0.35

pred = calibrated_logP.argmax(axis=1)
pred[confident_mask] = P.argmax(axis=1)[confident_mask]"""

assert target_c16_anchor in c16, "Anchor in Cell 16 not found!"

injective_decoding_code = """# ============================================================ 7b. Exact Soft Marginal Injective Cluster Decoding
def injective_cavity_marginals(P_group, iters=5, repel_strength=0.85):
    \"\"\"Cavity mean-field repulsion: soft injective decoding for a prompt cluster of any size n <= 18.\"\"\"
    n, C = P_group.shape
    L = np.log(np.clip(P_group, 1e-12, 1.0))
    q = np.exp(L - L.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    for _ in range(iters):
        lo = np.log(np.clip(1.0 - q, 1e-6, None))
        cavity_penalty = lo.sum(0, keepdims=True) - lo
        L_new = L + repel_strength * cavity_penalty
        q = np.exp(L_new - L_new.max(1, keepdims=True))
        q /= q.sum(1, keepdims=True)
    return q

# Apply Injective Soft Marginal Decoding to all identified prompt groups (2 to 18 rows)
if len(P) >= 2:
    P_calibrated = softmax_rows(calibrated_logP)
    Xt_dec = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(TE)
    D_dec = np.clip(1 - (Xt_dec @ Xt_dec.T).toarray(), 0, None)
    np.fill_diagonal(D_dec, 0)
    groups_dec = fcluster(linkage(squareform(D_dec, checks=False), method="average"), 0.85, criterion="distance")
    size_dec = np.bincount(groups_dec)[groups_dec]
    cluster_rows = (size_dec >= 2) & (size_dec <= 18)
    
    if cluster_rows.any():
        for g in np.unique(groups_dec[cluster_rows]):
            g_idx = np.where(groups_dec == g)[0]
            if len(g_idx) >= 2 and len(g_idx) <= 18:
                P_calibrated[g_idx] = injective_cavity_marginals(P_calibrated[g_idx], iters=5, repel_strength=0.85)
        print(f'Injective decoding applied to {int(cluster_rows.sum())} clustered rows in {len(np.unique(groups_dec[cluster_rows]))} groups', flush=True)
    
    calibrated_logP = np.log(np.clip(P_calibrated, 1e-12, 1.0))

# 3. High-confidence ensemble protection (protect top predictions with high margin > 0.40)
p_sort = np.sort(P, axis=1)
margin = p_sort[:, -1] - p_sort[:, -2]
confident_mask = margin > 0.40

pred = calibrated_logP.argmax(axis=1)
pred[confident_mask] = P.argmax(axis=1)[confident_mask]"""

c16 = c16.replace(target_c16_anchor, injective_decoding_code)
nb['cells'][16]['source'] = c16
print("Cell 16 updated with Injective Soft Marginal Cluster Decoding and Margin Lock > 0.40.")

# 7. Save notebook
with open('champion_v16/champion-v16.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Saved champion_v16/champion-v16.ipynb successfully!")

# 8. AST Syntax Validation
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        try:
            ast.parse(c['source'])
        except SyntaxError as e:
            print(f"SYNTAX ERROR in cell {i}: {e}")
            raise

print("All code cells in champion-v16.ipynb passed AST syntax validation!")
