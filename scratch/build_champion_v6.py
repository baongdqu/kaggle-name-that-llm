import json
import os

meta = {
  "id": "baonguyenduyquoc/name-that-llm-champion-v6-marginlock-top1",
  "title": "Name That LLM - Champion V6 MarginLock Top1",
  "code_file": "champion-v6.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": True,
  "enable_gpu": True,
  "enable_tpu": False,
  "enable_internet": False,
  "dataset_sources": [
    "bbucxi/qwen35-9b-safetensors",
    "pjmathematician/ultrabert-380m-ntl",
    "seyitkaangunes/ntl-extra-api-passages",
    "olegzholobov/name-that-llm-response-corpus"
  ],
  "kernel_sources": [
    "lucasboesen/name-that-llm-glm4-9b-trunc25",
    "lucasboesen/name-that-llm-extract-encoders2",
    "lucasboesen/name-that-llm-electra-large-weights",
    "lucasboesen/name-that-llm-desklib-weights",
    "lightningv08/ntl-lora24-fivefold-training",
    "lucasboesen/name-that-llm-train-lora24-s7",
    "lucasboesen/name-that-llm-train-tokattn",
    "lucasboesen/name-that-llm-train-tokattn2"
  ],
  "competition_sources": [
    "name-that-llm"
  ],
  "model_sources": [
    "qwen-lm/qwen-3/Transformers/4b-base/1",
    "qwen-lm/qwen-3/Transformers/8b-base/1",
    "qwen-lm/qwen2.5/Transformers/7b/1"
  ],
  "machine_shape": "NvidiaTeslaT4"
}

with open('champion_v6/kernel-metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)

nb = json.load(open('champion_v6/champion-v6.ipynb', encoding='utf-8'))
code = ''.join(nb['cells'][1]['source'])

# Replace the plain Sinkhorn section with Margin-Locked Soft Sinkhorn
old_sinkhorn = """P = np.exp(LOGIT - LOGIT.max(1, keepdims=True))
P /= P.sum(1, keepdims=True)
if BALANCE_ALPHA > 0 and len(P) >= 2:
    target = (1 - BALANCE_ALPHA) * P.sum(0) + BALANCE_ALPHA * np.full(K, len(P) / K)
    for _ in range(50):
        P *= (target / (P.sum(0) + 1e-12))[None, :]
        P /= P.sum(1, keepdims=True) + 1e-12
    print('balancing changed', int((P.argmax(1) != LOGIT.argmax(1)).sum()), 'of', len(P), flush=True)"""

margin_locked_sinkhorn = """P = np.exp(LOGIT - LOGIT.max(1, keepdims=True))
P /= P.sum(1, keepdims=True)

# ------------------------------------------------ Margin-Locked Soft Sinkhorn Balancing
# Lock 100% of high-confidence predictions to eliminate Hungarian-style corruption,
# and dynamically balance residual quota on malleable borderline samples only!
if BALANCE_ALPHA > 0 and len(P) >= 18:
    try:
        _top2_idx = np.argsort(P, axis=1)[:, -2:]
        _p_top1 = P[np.arange(len(P)), _top2_idx[:, 1]]
        _p_top2 = P[np.arange(len(P)), _top2_idx[:, 0]]
        _margins = _p_top1 - _p_top2
        
        MARGIN_THRESH = 0.35
        _anchor_mask = (_margins > MARGIN_THRESH) | (_p_top1 > 0.60)
        _n_anchors = int(_anchor_mask.sum())
        _n_malleable = len(P) - _n_anchors
        
        print(f"Margin-Lock: {len(P)} test rows -> {_n_anchors} locked anchors ({_n_anchors/len(P)*100:.1f}%), {_n_malleable} malleable rows", flush=True)
        
        if _n_malleable > 0:
            _locked_preds = P.argmax(axis=1)[_anchor_mask]
            _target_total = np.full(K, len(P) / K)
            _locked_counts = np.bincount(_locked_preds, minlength=K)
            _residual_target = np.maximum(1.0, _target_total - _locked_counts)
            _residual_target = _residual_target / _residual_target.sum() * _n_malleable
            
            _P_mal = P[~_anchor_mask].copy()
            for _ in range(50):
                _P_mal *= (_residual_target / (_P_mal.sum(axis=0) + 1e-12))[None, :]
                _P_mal /= _P_mal.sum(axis=1, keepdims=True) + 1e-12
            
            P[~_anchor_mask] = _P_mal
            print('Residual Sinkhorn successfully balanced malleable predictions toward optimal macro-recall!', flush=True)
    except Exception as e:
        print('Margin-locked Sinkhorn fallback:', e, flush=True)
elif BALANCE_ALPHA > 0 and len(P) >= 2:
    target = (1 - BALANCE_ALPHA) * P.sum(0) + BALANCE_ALPHA * np.full(K, len(P) / K)
    for _ in range(50):
        P *= (target / (P.sum(0) + 1e-12))[None, :]
        P /= P.sum(1, keepdims=True) + 1e-12
    print('balancing changed', int((P.argmax(1) != LOGIT.argmax(1)).sum()), 'of', len(P), flush=True)"""

assert old_sinkhorn in code, "old_sinkhorn not found in code!"
code = code.replace(old_sinkhorn, margin_locked_sinkhorn)

nb['cells'][1]['source'] = code
with open('champion_v6/champion-v6.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("SUCCESS: champion-v6.ipynb created successfully!")
