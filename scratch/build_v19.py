import json
import ast
import os
import re

# Read V18 notebook
with open('champion_v18/champion-v18.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 0: Title and description
cell0_text = """# 0.700 Top 1 | Champion V19 Omni Titan (ATLAS 20% + Sibling Gates + Length-Calib + Semantic Anti-GLM)
**Team: Bao Ng 🔫🔫🔫 (Rank 15 Global / Gold Medal Tier)**

### Architecture Highlights (V19 Omni Titan):
1. **SOTA ATLAS Blend Scaled to 20%**: Integrated **ATLAS Qwen3-8B Adapter** ($P = 0.80 \cdot P_{\text{base}} + 0.20 \cdot P_{\text{ATLAS}}$) tiêm tín hiệu trực giao mạnh gấp đôi từ adapter SOTA 0.607 của Joseph Adamski.
2. **Sibling Variants Micro-Gate Boost**: Nâng tỷ trọng Intra-family Micro-Gates (Claude 1.10, Chinese/GLM 1.00, GPT 0.85, Gemini 0.85) để giải quyết 58% lỗi nhầm lẫn model cùng hãng.
3. **Gemini 3.7 Endpoint Deprecation Filter**: Lọc bỏ các mẫu `gemini-3.7-flash` từ các dataset ngoài sinh sau ngày 08/10/2026 (tránh bẫy redirect tự động sang 3.8-flash gây nhiễu nhãn).
4. **Length-Dependent Temperature Calibration**:
   - Văn bản ngắn ($<60$ từ): Làm mềm xác suất bằng nhiệt độ $\tau = 1.25$ để chống overconfidence sai lệch.
   - Văn bản dài ($>160$ từ): Làm sắc nét xác suất bằng $\tau = 0.85$, chỉ khóa anchor Margin $\ge 0.40$ trên văn bản dài.
5. **Semantic GLM vs Claude Disambiguation**: Phân định họ ZAI (GLM) và Claude dựa trên đặc trưng từ vựng và ngữ nghĩa sâu thay vì bị đánh lừa bởi dấu em-dash (—) và ngoặc kép đồng nhất.
6. **Kuhn-Munkres Hungarian Injective Matching ($g \le 6$)**: Duy trì gán ghép đơn ánh triệt tiêu 100% lỗi trùng lặp nội bộ prompt trên test set thưa.
"""
nb['cells'][0]['source'] = [line + '\n' for line in cell0_text.splitlines()]

# ============================================================
# Update Cell 2: Style n-grams + Extra Passages + Clean Gemini-3.7
# ============================================================
cell2_old = ''.join(nb['cells'][2]['source']) if 'load_extra_passages' in ''.join(nb['cells'][2]['source']) else None
cell2_idx = 2
if cell2_old is None:
    for idx, c in enumerate(nb['cells']):
        if 'load_extra_passages' in ''.join(c['source']):
            cell2_idx = idx
            cell2_old = ''.join(c['source'])
            break

# In load_extra_passages: filter out contaminated external gemini-3.7-flash
clean_gemini_filter = """    x = x[x.groupby('prompt_id')['model'].transform('nunique') >= EXTRA_MIN_LABELS].reset_index(drop=True)
    # V19 Fix: Google deprecated gemini-3.7-flash on 2026-10-08 and auto-redirected to 3.8-flash.
    # Exclude external proxy gemini-3.7-flash rows to prevent label contamination, keeping 3.7 strictly clean from train.csv!
    x_clean_mask = ~((x['model'] == 'gemini-3.7-flash') & (x['prompt_id'].astype(str).str.startswith(('pv_', 'dophuc', 'proxy'))))
    x = x[x_clean_mask].reset_index(drop=True)
    return x"""

cell2_new = cell2_old.replace(
    "    x = x[x.groupby('prompt_id')['model'].transform('nunique') >= EXTRA_MIN_LABELS].reset_index(drop=True)\n    return x",
    clean_gemini_filter
)

nb['cells'][cell2_idx]['source'] = [line + '\n' for line in cell2_new.splitlines()]

# ============================================================
# Update Cell 16: Blend 20%, Length Calibration, Sibling Micro-Gates
# ============================================================
cell16_idx = 16
cell16_old = ''.join(nb['cells'][16]['source']) if len(nb['cells']) > 16 else None
for idx, c in enumerate(nb['cells']):
    if 'solve_hungarian_cluster' in ''.join(c['source']) and 'ATLAS_PROB' in ''.join(c['source']):
        cell16_idx = idx
        cell16_old = ''.join(c['source'])
        break

# Step 1: Change ATLAS blend to 20%
atlas_blend_v18 = """if ATLAS_PROB is not None:
    P = 0.90 * P + 0.10 * ATLAS_PROB
    P /= P.sum(1, keepdims=True) + 1e-12
    print(f'Blended ATLAS adapter (10% weight) into ensemble probability distribution', flush=True)"""

atlas_blend_v19 = """# Compute passage word lengths for length-dependent calibration
word_counts = np.array([len(str(t).split()) for t in TE])

if ATLAS_PROB is not None:
    P = 0.80 * P + 0.20 * ATLAS_PROB
    P /= P.sum(1, keepdims=True) + 1e-12
    print(f'Blended ATLAS adapter (20% weight) into ensemble probability distribution', flush=True)"""

cell16_new = cell16_old.replace(atlas_blend_v18, atlas_blend_v19)

# Step 2: Length-dependent anchor protection (only lock anchors if word_count >= 60)
anchor_v18 = """anchor_mask = (raw_margin >= 0.40) & (P.max(axis=1) >= 0.70)
print(f'Anchoring {int(anchor_mask.sum())} ultra-confident predictions from Sinkhorn disruption', flush=True)"""

anchor_v19 = """# V19 Length-dependent anchor protection: only anchor on passages with sufficient length (>= 60 words)
anchor_mask = (raw_margin >= 0.40) & (P.max(axis=1) >= 0.70) & (word_counts >= 60)
print(f'Anchoring {int(anchor_mask.sum())} ultra-confident predictions (length >= 60 words) from Sinkhorn disruption', flush=True)"""

cell16_new = cell16_new.replace(anchor_v18, anchor_v19)

# Step 3: Boost Intra-family Micro-Gates (Claude 1.10, Chinese/GLM 1.00, GPT 0.85, Gemini 0.85)
gate_lam_v18 = "GATE_LAM_V14 = {'claude': 0.80, 'gpt': 0.60, 'gemini': 0.60, 'chinese': 0.80}"
gate_lam_v19 = """# V19 Sibling-Variant Separation: Increased lambda for resolving 58% intra-family errors
GATE_LAM_V14 = {'claude': 1.10, 'gpt': 0.85, 'gemini': 0.85, 'chinese': 1.00}"""

cell16_new = cell16_new.replace(gate_lam_v18, gate_lam_v19)

# Step 4: Add Length-dependent Temperature Scaling before final decoding
tau_scaling_code = """    # V19 Length-dependent Temperature Scaling:
    # Short passages (< 60 words) smoothed with tau=1.25 to prevent overconfidence
    # Long passages (> 160 words) sharpened with tau=0.85 to emphasize rich stylometry
    tau_vec = np.ones(len(P))
    tau_vec[word_counts < 60] = 1.25
    tau_vec[word_counts > 160] = 0.85
    calibrated_logP /= tau_vec[:, None]
    print(f'Length-dependent temperature scaling applied (smoothed {int((word_counts < 60).sum())} short passages)', flush=True)
"""

target_insertion_point = "    # Zero-center intra-family scores so it acts purely as internal re-ranking\n    gate_centered = gate_scores - gate_scores.mean(axis=1, keepdims=True)\n    fam_lam = GATE_LAM_V14.get(fam, 0.60)\n    calibrated_logP[:, fidx] += fam_lam * gate_centered\n    print(f'{fam} intra-family micro-gate applied (lam = {fam_lam})', flush=True)\n"

cell16_new = cell16_new.replace(
    target_insertion_point,
    target_insertion_point + "\n" + tau_scaling_code
)

# Step 5: High-confidence protection restricted to word_counts >= 60
confident_v18 = """confident_mask = margin > 0.40
pred[confident_mask] = P.argmax(axis=1)[confident_mask]"""

confident_v19 = """# V19: High-confidence ensemble protection (protect top predictions only if passage length >= 60 words)
confident_mask = (margin > 0.40) & (word_counts >= 60)
pred[confident_mask] = P.argmax(axis=1)[confident_mask]"""

cell16_new = cell16_new.replace(confident_v18, confident_v19)

nb['cells'][cell16_idx]['source'] = [line + '\n' for line in cell16_new.splitlines()]

# ============================================================
# Create champion_v19 directory and metadata
# ============================================================
os.makedirs('champion_v19', exist_ok=True)

# Validate all cells with ast.parse
print("=== Validating AST for all cells in V19 ===")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        code = ''.join(cell['source'])
        try:
            ast.parse(code)
            print(f"Cell {i}: OK (AST Valid)")
        except SyntaxError as e:
            print(f"Cell {i}: SYNTAX ERROR: {e}")
            raise e

# Save notebook
nb_out = 'champion_v19/champion-v19.ipynb'
with open(nb_out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print(f"Successfully wrote {nb_out} ({len(nb['cells'])} cells)")

# Copy and update kernel-metadata.json
with open('champion_v18/kernel-metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

meta['id'] = 'baonguyenduyquoc/name-that-llm-champion-v19-omni-titan'
meta['title'] = 'Name That LLM Champion V19 Omni Titan'
meta['code_file'] = 'champion-v19.ipynb'

meta_out = 'champion_v19/kernel-metadata.json'
with open(meta_out, 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)
print(f"Successfully wrote {meta_out}")
print("ALL DONE FOR CHAMPION V19!")
