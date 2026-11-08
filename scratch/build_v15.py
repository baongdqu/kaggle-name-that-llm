import json
import os
import ast

os.makedirs('champion_v15', exist_ok=True)

# 1. Metadata for V15 Grand Fusion
meta_v15 = {
  "id": "baonguyenduyquoc/name-that-llm-champion-v15-grand-fusion",
  "title": "Name That LLM Champion V15 Grand Fusion",
  "code_file": "champion-v15.ipynb",
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

with open('champion_v15/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta_v15, f, indent=2)

print("Saved champion_v15/kernel-metadata.json")

# 2. Load base notebook V14
nb = json.load(open('champion_v14/champion-v14.ipynb', encoding='utf-8'))

# 3. Update Cell 1: Add NISHANT_LABELS and permutation alignment mapping
c1 = nb['cells'][1]['source']
target_c1_anchor = "FAMILY_IDX = {"
assert target_c1_anchor in c1, "Anchor in Cell 1 not found!"

nishant_mapping_code = """# Nishant Dahal Two-Stage LoRA label ordering and alignment mapping to V15
NISHANT_LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                  'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                  'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
                  'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
NISHANT_TO_V14 = np.array([LABELS.index(l) for l in NISHANT_LABELS])

"""

c1 = c1.replace(target_c1_anchor, nishant_mapping_code + target_c1_anchor)
nb['cells'][1]['source'] = c1
print("Cell 1 updated with Nishant label mapping.")

# 4. Update Cell 12: Add score_two_stage_loras block
c12 = nb['cells'][12]['source']
target_c12_anchor = """if not SMOKE:
    for trunk_spec, pool_layers, checkpoints in LORA_RUNS:
        score_lora_checkpoints(trunk_spec, pool_layers, checkpoints)"""

assert target_c12_anchor in c12, "Anchor in Cell 12 not found!"

ts_lora_code = """if not SMOKE:
    for trunk_spec, pool_layers, checkpoints in LORA_RUNS:
        score_lora_checkpoints(trunk_spec, pool_layers, checkpoints)


# ============================================================ 5b. Nishant Dahal Two-Stage LoRAs (Qwen3-14B & Qwen3-8B)
def score_two_stage_loras():
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from safetensors.torch import load_file
    
    def find_file(pattern):
        hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
        return hits[0] if hits else None
    
    def find_base_spec(spec):
        for cfg in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
            try:
                c = json.load(open(cfg))
            except Exception:
                continue
            if all(c.get(k) == v for k, v in spec.items()):
                return os.path.dirname(cfg)
        return None

    def load_ts_trunk(base_dir, keep, dtype):
        mc = AutoConfig.from_pretrained(base_dir)
        mc.num_hidden_layers, mc.use_cache = keep, False
        if getattr(mc, "layer_types", None) is not None:
            mc.layer_types = mc.layer_types[:keep]
        hd = getattr(mc, "head_dim", mc.hidden_size // mc.num_attention_heads)
        gib = 2 * (mc.vocab_size * mc.hidden_size + keep * (2 * mc.hidden_size * mc.num_attention_heads * hd
                   + 2 * mc.hidden_size * mc.num_key_value_heads * hd + 3 * mc.hidden_size * mc.intermediate_size)) / 2 ** 30
        target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if gib > 13 and torch.cuda.device_count() > 1:
            mem = {i: "13GiB" for i in range(torch.cuda.device_count())}
            model = AutoModel.from_pretrained(base_dir, config=mc, dtype=dtype, device_map="auto", max_memory=mem).eval()
        else:
            model = AutoModel.from_pretrained(base_dir, config=mc, dtype=dtype).to(target_device).eval()
        model.norm = torch.nn.Identity()
        return model

    @torch.no_grad()
    def merge_ts_lora(model, adapter_dir):
        acfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
        scale = acfg["lora_alpha"] / acfg["r"]
        sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
        modules = dict(model.named_modules())
        for k, A in sd.items():
            if ".lora_A." in k:
                name = k.split(".lora_A.")[0]
                lin = modules[name[name.index("layers."):]]
                Bm = sd[k.replace(".lora_A.", ".lora_B.")]
                lin.weight.add_(((Bm.float().to(lin.weight.device) @ A.float().to(lin.weight.device)) * scale).to(lin.weight.dtype))

    @torch.no_grad()
    def run_ts_lora(ckpt_dir, base_dir, texts, bs):
        meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))
        assert meta.get("labels") == NISHANT_LABELS, "Nishant Dahal label order check failed!"
        cfg = meta["cfg"]
        target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else (torch.float16 if torch.cuda.is_available() else torch.float32)
        tok = AutoTokenizer.from_pretrained(base_dir)
        tok.padding_side = "right"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = load_ts_trunk(base_dir, cfg["keep"], dtype)
        merge_ts_lora(model, os.path.join(ckpt_dir, "adapter"))
        heads = load_file(os.path.join(ckpt_dir, "heads.safetensors"))
        layers, H = cfg["layers"], model.config.hidden_size
        norms = torch.nn.ModuleList([torch.nn.LayerNorm(H) for _ in layers]).to(target_device).float()
        norms.load_state_dict({k[len("norms."):]: v for k, v in heads.items() if k.startswith("norms.")})
        ids = tok(list(texts), add_special_tokens=False, truncation=True, max_length=cfg["max_len"])["input_ids"]
        ids = [x + [tok.eos_token_id] for x in ids]
        order = np.argsort([len(x) for x in ids], kind="stable")
        Z = np.zeros((len(ids), H * len(layers)), np.float32)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(model.embed_tokens.weight.device)
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            if not all(torch.isfinite(hs[l]).all() for l in layers):
                model.to(torch.bfloat16)
                hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
                model.to(dtype)
            m = enc["attention_mask"].unsqueeze(-1).float().to(target_device)
            m[:, 0] = 0
            den = m.sum(1).clamp(min=1)
            Z[idx] = torch.cat([nm((hs[l].to(target_device).float() * m).sum(1) / den) for l, nm in zip(layers, norms)], -1).cpu().numpy()
        del model
        torch.cuda.empty_cache()
        return Z, {k: v.float().numpy() for k, v in heads.items() if not k.startswith("norms.")}

    TS_LORAS = {
        "lora14b_v1": dict(w=0.35, twin_w=0.12, bs=8,  base=dict(model_type="qwen3", hidden_size=5120, num_hidden_layers=40)),
        "lora8b_v1":  dict(w=0.20, twin_w=0.08, bs=16, base=dict(model_type="qwen3", hidden_size=4096, num_hidden_layers=36)),
    }

    # Prompt topic clusters for centred twin
    Xt = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(TE)
    D = np.clip(1 - (Xt @ Xt.T).toarray(), 0, None)
    np.fill_diagonal(D, 0)
    groups = fcluster(linkage(squareform(D, checks=False), method="average"), 0.85, criterion="distance") if len(TE) >= 2 else np.ones(len(TE), dtype=int)
    size = np.bincount(groups)[groups]
    twin_rows = (size >= 2) & (size <= 18)
    del D, Xt
    print(f"Two-Stage LoRA topic clusters: {int((size == 1).sum())} singletons, {int(twin_rows.sum())} in clusters", flush=True)

    for ckpt_name, mc in TS_LORAS.items():
        meta_file = find_file(f"{ckpt_name}/meta.json")
        if not meta_file:
            print(f"Two-stage LoRA checkpoint {ckpt_name} not attached, skipping", flush=True)
            continue
        base_dir = find_base_spec(mc["base"])
        if not base_dir:
            print(f"Base model for {ckpt_name} not found, skipping", flush=True)
            continue
        ckpt_dir = os.path.dirname(meta_file)
        print(f"Scoring Two-Stage LoRA: {ckpt_name}...", flush=True)
        Z, heads = run_ts_lora(ckpt_dir, base_dir, TE, mc["bs"])
        
        # Plain prediction mapped to V15 alphabetical label order
        prob_plain_nishant = softmax_rows(Z @ heads["plain.weight"].T + heads["plain.bias"])
        prob_plain_v15 = np.zeros_like(prob_plain_nishant)
        prob_plain_v15[:, NISHANT_TO_V14] = prob_plain_nishant
        LOGIT[:] += mc["w"] * np.log(np.clip(prob_plain_v15, 1e-9, 1.0))
        print(f"  + {ckpt_name} plain head added (weight {mc['w']})", flush=True)

        # Centred twin prediction on grouped rows
        if twin_rows.any():
            Zc = Z.copy()
            for g in np.unique(groups[twin_rows]):
                Zc[groups == g] -= Zc[groups == g].mean(0)
            centred = Zc[twin_rows] @ heads["centered.weight"].T + heads["centered.bias"]
            prob_centred_nishant = softmax_rows(centred)
            prob_centred_v15 = np.zeros_like(prob_centred_nishant)
            prob_centred_v15[:, NISHANT_TO_V14] = prob_centred_nishant
            LOGIT[twin_rows] += mc["twin_w"] * np.log(np.clip(prob_centred_v15, 1e-9, 1.0))
            print(f"  + {ckpt_name} centred twin added on {int(twin_rows.sum())} rows (weight {mc['twin_w']})", flush=True)
        del Z
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Finished {ckpt_name} ({time.time() - T0:.0f}s)", flush=True)


if not SMOKE:
    score_two_stage_loras()"""

c12 = c12.replace(target_c12_anchor, ts_lora_code)
nb['cells'][12]['source'] = c12
print("Cell 12 updated with score_two_stage_loras.")

# 5. Title and intro update
nb['cells'][0]['source'] = """# 0.605+ | Champion V15 Grand Fusion (Two-Stage 14B LoRA + In-Style Mega Gate + Gaussian Group Heads)

**Champion V15 Pipeline**:
- **Branch A (Deep 14B Semantics)**: Nishant Dahal's Two-Stage Qwen3-14B Base (27 layers, hidden 5120) & Qwen3-8B Base (24 layers, hidden 4096) trained on 35k passages from `nishantdahal/ntl-two-stage-lora`
- **Branch B (Stylometrics & Clustered Repulsion)**: Champion V14 (0.577 Rank 33):
  - In-Style Qwen3-8B LoRA + Qwen2.5-7B LoRA + LoRA24 Public, Seed7, Claude-head
  - StyleGrams 4 Views trained on Train + 9,818 In-Style Passages from 709 prompt banks
  - Shared-Prompt Gaussian Group Heads on 4 dense backbone views with At-Most-Once pairwise repulsion
  - Token-Attention Heads (Qwen3.5-9B, DeBERTa-v3-large) + UltraBERT-380M
- **Marginal Balancing**: Sinkhorn transport balancing (alpha 0.5)
- **Gating**: Universal 18-Class Mega-Gate + 4-Way Chinese Frontier Gate with Cross-Family Soft Posterior Calibration & High Confidence Margin Lock
"""

# 6. Save notebook
with open('champion_v15/champion-v15.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Saved champion_v15/champion-v15.ipynb successfully!")

# 7. Syntax validation
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        try:
            ast.parse(c['source'])
        except SyntaxError as e:
            print(f"SYNTAX ERROR in cell {i}: {e}")
            raise

print("All code cells in champion-v15.ipynb passed AST syntax validation!")
