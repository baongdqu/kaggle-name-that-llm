import json

cells = []

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")[:-1]] + [text.split("\n")[-1]]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")[:-1]] + [code.split("\n")[-1]]
    }

# ----------------- CELL 0 -----------------
cells.append(md_cell("""# 🏆 Name That LLM: Model Arena Benchmark (Offline Multi-Model Evaluation)
## Empirical Head-to-Head Comparison of 6 SOTA LLM Backbones on Unseen Prompt GroupKFold Holdout
### Evaluates: ModernBERT-large, DeBERTa-v3-large, Qwen3-14B, Qwen3-8B, Google Gemma-2-9B, and 4-View Stylometrics
**Objectives:**
1. **Solo Power:** Measure pure standalone accuracy & log-loss per model.
2. **Orthogonality:** Calculate pairwise error disagreement rates to find true diversity.
3. **Optimal Ensemble Weights:** Run greedy forward selection to find the optimal fusion recipe."""))

# ----------------- CELL 1 -----------------
cells.append(code_cell("""import os, sys, time, gc, glob, json, re, shutil
import numpy as np, pandas as pd, torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from scipy.sparse import hstack
from transformers import AutoTokenizer, AutoModel, AutoConfig
from safetensors.torch import load_file

print(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", flush=True)

# 1. Locate Datasets and Models
DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
train_df = pd.read_csv(DATA + '/train.csv')
print(f"Loaded train.csv with {len(train_df)} samples across {train_df.model.nunique()} models and {train_df.prompt_id.nunique()} prompt groups.")

LABELS = sorted(train_df.model.unique())
K = len(LABELS)
L2I = {l: i for i, l in enumerate(LABELS)}
y_all = train_df.model.map(L2I).values
prompts_all = train_df.prompt_id.values

# 2. Fixed GroupKFold Split: Hold out ~540 unseen prompt samples for the Arena test
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(train_df, y_all, prompts_all))

tr_df = train_df.iloc[train_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

y_val = val_df.model.map(L2I).values
val_texts = val_df.text.astype(str).tolist()
val_groups = val_df.prompt_id.values

print(f"Arena Train Size: {len(tr_df)} (Prompts: {tr_df.prompt_id.nunique()})")
print(f"Arena Validation Holdout: {len(val_df)} (Prompts: {val_df.prompt_id.nunique()}, Unseen in Train)")

def log_softmax(L):
    m = np.max(L, axis=1, keepdims=True)
    return L - m - np.log(np.sum(np.exp(L - m), axis=1, keepdims=True))

MODEL_PREDS = {}
MODEL_TIMES = {}
"""))

# ----------------- CELL 2 -----------------
cells.append(md_cell("""### Model 1: 4-View Stylometrics (LinearSVC on Char + Word + Shape + Small Words)"""))

cells.append(code_cell("""print("=== EVALUATING MODEL 1: 4-VIEW STYLOMETRICS ===", flush=True)
t0 = time.time()

SMALL_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())

def shape_text(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", str(t))

def small_words_text(t):
    return " ".join(w.lower() if w.lower() in SMALL_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", str(t)))

tr_texts = tr_df.text.astype(str).tolist()
y_tr = tr_df.model.map(L2I).values

A_tr, A_va = [], []
VIEWS = [
    ("char",  3, 5, 250000, 1.0, tr_texts, val_texts, 'char', None),
    ("word",  1, 2, 180000, 0.7, tr_texts, val_texts, 'word', r"(?u)\b\w+\b|[^\w\s]"),
    ("shape", 2, 5,  60000, 0.5, [shape_text(t) for t in tr_texts], [shape_text(t) for t in val_texts], 'char', None),
    ("small", 1, 3,  50000, 0.5, [small_words_text(t) for t in tr_texts], [small_words_text(t) for t in val_texts], 'word', None),
]

for name, mn, mx, max_feat, w, x_tr, x_va, analyzer, tok_pat in VIEWS:
    kw = dict(ngram_range=(mn, mx), analyzer=analyzer, min_df=2, max_features=max_feat, sublinear_tf=True)
    if tok_pat: kw['token_pattern'] = tok_pat
    vec = TfidfVectorizer(**kw)
    A_tr.append(vec.fit_transform(x_tr) * w)
    A_va.append(vec.transform(x_va) * w)

clf = LinearSVC(C=0.5, random_state=42, max_iter=3000, class_weight='balanced')
clf.fit(hstack(A_tr, format='csr'), y_tr)
svc_logits = clf.decision_function(hstack(A_va, format='csr'))
svc_probs = np.exp(log_softmax(svc_logits))

acc_svc = accuracy_score(y_val, svc_probs.argmax(axis=1))
MODEL_PREDS['Stylometrics_4View'] = svc_probs
MODEL_TIMES['Stylometrics_4View'] = time.time() - t0
print(f"Model 1 (Stylometrics 4-View): Accuracy = {acc_svc*100:.2f}%, Elapsed = {MODEL_TIMES['Stylometrics_4View']:.1f}s", flush=True)
del A_tr, A_va, clf; gc.collect()
"""))

# ----------------- CELL 3 -----------------
cells.append(md_cell("""### Model 2: ModernBERT-large (8,192 Context Bi-directional Transformer)"""))

cells.append(code_cell("""print("=== EVALUATING MODEL 2: MODERNBERT-LARGE ===", flush=True)
t0 = time.time()

def find_enc_dir(name):
    hits = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/meta.json', recursive=True) if f'/{name}/' in p or (name == 'mb_v5' and 'nlm-enc-v5' in p)]
    return hits[0] if hits else None

MB_PATH = find_enc_dir('mb_v5')
print(f"ModernBERT directory found: {MB_PATH}", flush=True)

if MB_PATH and os.path.exists(os.path.join(MB_PATH, 'head.safetensors')):
    meta = json.load(open(os.path.join(MB_PATH, 'meta.json')))
    perm = [meta['labels'].index(l) for l in LABELS]
    head = load_file(os.path.join(MB_PATH, 'head.safetensors'))
    W = head['weight'].float().numpy()[perm]
    b = head['bias'].float().numpy()[perm]
    
    etok = AutoTokenizer.from_pretrained(os.path.join(MB_PATH, 'encoder'))
    enc = AutoModel.from_pretrained(os.path.join(MB_PATH, 'encoder'), dtype=torch.float32).to('cuda:0').eval()
    
    mb_feats = []
    with torch.no_grad():
        for i in range(0, len(val_texts), 16):
            batch = val_texts[i:i+16]
            toks = etok(batch, padding=True, truncation=True, max_length=1024, return_tensors='pt').to('cuda:0')
            hs = enc(**toks).last_hidden_state
            mask = toks['attention_mask'].unsqueeze(-1).float()
            feat = (hs * mask).sum(1) / mask.sum(1)
            mb_feats.append(feat.cpu().numpy())
    
    mb_feats = np.concatenate(mb_feats, axis=0)
    mb_logits = mb_feats @ W.T + b
    mb_probs = np.exp(log_softmax(mb_logits))
    acc_mb = accuracy_score(y_val, mb_probs.argmax(axis=1))
    MODEL_PREDS['ModernBERT_large'] = mb_probs
    MODEL_TIMES['ModernBERT_large'] = time.time() - t0
    print(f"Model 2 (ModernBERT-large): Accuracy = {acc_mb*100:.2f}%, Elapsed = {MODEL_TIMES['ModernBERT_large']:.1f}s", flush=True)
    del enc, etok, mb_feats; gc.collect(); torch.cuda.empty_cache()
else:
    print("ModernBERT-large checkpoint not found, skipping.", flush=True)
"""))

# ----------------- CELL 4 -----------------
cells.append(md_cell("""### Model 3: DeBERTa-v3-large (Microsoft Syntax & Disentangled Attention Specialist)"""))

cells.append(code_cell("""print("=== EVALUATING MODEL 3: DEBERTA-V3-LARGE ===", flush=True)
t0 = time.time()

DEB_PATH = find_enc_dir('deberta_v3_large')
print(f"DeBERTa directory found: {DEB_PATH}", flush=True)

if DEB_PATH and os.path.exists(os.path.join(DEB_PATH, 'head.safetensors')):
    meta = json.load(open(os.path.join(DEB_PATH, 'meta.json')))
    perm = [meta['labels'].index(l) for l in LABELS]
    head = load_file(os.path.join(DEB_PATH, 'head.safetensors'))
    W = head['weight'].float().numpy()[perm]
    b = head['bias'].float().numpy()[perm]
    
    etok = AutoTokenizer.from_pretrained(os.path.join(DEB_PATH, 'encoder'))
    enc = AutoModel.from_pretrained(os.path.join(DEB_PATH, 'encoder'), dtype=torch.float32).to('cuda:0').eval()
    
    deb_feats = []
    with torch.no_grad():
        for i in range(0, len(val_texts), 16):
            batch = val_texts[i:i+16]
            toks = etok(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda:0')
            hs = enc(**toks).last_hidden_state
            mask = toks['attention_mask'].unsqueeze(-1).float()
            feat = (hs * mask).sum(1) / mask.sum(1)
            deb_feats.append(feat.cpu().numpy())
            
    deb_feats = np.concatenate(deb_feats, axis=0)
    deb_logits = deb_feats @ W.T + b
    deb_probs = np.exp(log_softmax(deb_logits))
    acc_deb = accuracy_score(y_val, deb_probs.argmax(axis=1))
    MODEL_PREDS['DeBERTa_v3_large'] = deb_probs
    MODEL_TIMES['DeBERTa_v3_large'] = time.time() - t0
    print(f"Model 3 (DeBERTa-v3-large): Accuracy = {acc_deb*100:.2f}%, Elapsed = {MODEL_TIMES['DeBERTa_v3_large']:.1f}s", flush=True)
    del enc, etok, deb_feats; gc.collect(); torch.cuda.empty_cache()
else:
    print("DeBERTa-v3-large checkpoint not found, skipping.", flush=True)
"""))

# ----------------- CELL 5 -----------------
cells.append(md_cell("""### General Function for LoRA24 Evaluation on Truncated Causal Backbones"""))

cells.append(code_cell("""@torch.no_grad()
def evaluate_lora_backbone(name, model_path, adapter_path, keep_layers, layers_list=[2, 4, 6, 8, 12, 16, 20, 24], max_len=384, bs=8):
    if not model_path or not os.path.exists(model_path):
        print(f"Base model for {name} not found at {model_path}, skipping.", flush=True)
        return None
    if not adapter_path or not os.path.exists(adapter_path):
        print(f"Adapter for {name} not found at {adapter_path}, skipping.", flush=True)
        return None
        
    print(f"=== EVALUATING {name.upper()} ===", flush=True)
    t_start = time.time()
    
    tok = AutoTokenizer.from_pretrained(model_path)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    cfg = AutoConfig.from_pretrained(model_path)
    tc = getattr(cfg, 'text_config', cfg)
    tc.num_hidden_layers = keep_layers
    tc.use_cache = False
    if getattr(tc, 'layer_types', None) is not None:
        tc.layer_types = tc.layer_types[:keep_layers]
        
    m = AutoModel.from_pretrained(model_path, config=cfg, dtype=torch.float16, device_map='auto').eval()
    mods = dict(m.named_modules())
    dev = next(m.parameters()).device
    
    # Merge adapter safely using .data.add_ to bypass autograd leaf variable checks
    adp = os.path.join(adapter_path, 'adapter') if os.path.exists(os.path.join(adapter_path, 'adapter', 'adapter_config.json')) else adapter_path
    acfg = json.load(open(os.path.join(adp, 'adapter_config.json')))
    scale = acfg['lora_alpha'] / acfg['r']
    sd = load_file(os.path.join(adp, 'adapter_model.safetensors'))
    for k, A in sd.items():
        if '.lora_A.' not in k: continue
        mname = k.replace('base_model.model.', '', 1).split('.lora_A.')[0]
        if mname not in mods and 'model.' + mname in mods: mname = 'model.' + mname
        if mname in mods:
            W = mods[mname].weight
            B = sd[k.replace('.lora_A.', '.lora_B.')]
            delta = ((B.to(W.device, torch.float32) @ A.to(W.device, torch.float32)) * scale).to(W.dtype)
            W.data.add_(delta)
            
    # Load heads
    H = torch.load(os.path.join(adapter_path, 'heads.pt'), map_location='cpu') if os.path.exists(os.path.join(adapter_path, 'heads.pt')) \
        else dict(load_file(os.path.join(adapter_path, 'heads.safetensors')))
    if os.path.exists(os.path.join(adapter_path, 'meta.json')):
        order = [json.load(open(os.path.join(adapter_path, 'meta.json')))['labels'].index(l) for l in LABELS]
        for h in ('plain', 'centered'):
            H[f'{h}.weight'], H[f'{h}.bias'] = H[f'{h}.weight'][order], H[f'{h}.bias'][order]
            
    hid = H['norms.0.weight'].shape[0]
    norms = [torch.nn.LayerNorm(hid) for _ in layers_list]
    for i, nm in enumerate(norms):
        nm.load_state_dict({'weight': H[f'norms.{i}.weight'], 'bias': H[f'norms.{i}.bias']})
        
    # Featurize
    enc_ids = tok(val_texts, truncation=True, max_length=max_len)['input_ids']
    order = np.argsort([len(x) for x in enc_ids])
    Z = np.zeros((len(enc_ids), hid * len(layers_list)), np.float32)
    
    for i in range(0, len(order), bs):
        b = order[i:i + bs]
        L_b = max(len(enc_ids[j]) for j in b)
        b_ids = torch.full((len(b), L_b), pad, dtype=torch.long)
        b_att = torch.zeros((len(b), L_b), dtype=torch.long)
        for r, j in enumerate(b):
            b_ids[r, :len(enc_ids[j])] = torch.tensor(enc_ids[j])
            b_att[r, :len(enc_ids[j])] = 1
            
        hs = m(input_ids=b_ids.to(dev), attention_mask=b_att.to(dev), output_hidden_states=True, use_cache=False).hidden_states
        if not all(torch.isfinite(hs[l]).all() for l in layers_list):
            m.to(torch.bfloat16)
            hs = m(input_ids=b_ids.to(dev), attention_mask=b_att.to(dev), output_hidden_states=True, use_cache=False).hidden_states
            m.to(torch.float16)
        mask = b_att.unsqueeze(-1).float(); mask[:, 0] = 0
        Z[b] = torch.cat([nm((hs[l].float().cpu() * mask).sum(1) / mask.sum(1)) for l, nm in zip(layers_list, norms)], -1).numpy()
        
    # Compute plain and centered logits
    head_fn = lambda h, X: log_softmax(X @ H[f'{h}.weight'].float().numpy().T + H[f'{h}.bias'].float().numpy())
    plain_logits = head_fn('plain', Z)
    
    Zc = Z.copy()
    for g in np.unique(val_groups):
        Zc[val_groups == g] -= Zc[val_groups == g].mean(0)
    cent_logits = head_fn('centered', Zc)
    
    final_logits = 0.65 * plain_logits + 0.35 * cent_logits
    final_probs = np.exp(log_softmax(final_logits))
    
    acc = accuracy_score(y_val, final_probs.argmax(axis=1))
    elapsed = time.time() - t_start
    print(f"{name}: Accuracy = {acc*100:.2f}%, Elapsed = {elapsed:.1f}s", flush=True)
    
    del m, mods, sd, H, Z, Zc; gc.collect(); torch.cuda.empty_cache()
    return final_probs, elapsed
"""))

# ----------------- CELL 6 -----------------
cells.append(md_cell("""### Model 4 & 5: Qwen3-8B and Qwen3-14B LoRA24"""))

cells.append(code_cell("""def model_dir_opt(*parts):
    hits = sorted(os.path.dirname(c) for c in glob.glob('/kaggle/input/**/config.json', recursive=True) if all(p in c.lower() for p in parts))
    return hits[0] if hits else None

def adapter_dir(tag):
    hits = [os.path.dirname(p) for p in glob.glob(f'/kaggle/input/**/{tag}/heads.*', recursive=True)]
    if not hits:
        hits = glob.glob(f'/kaggle/input/**/{tag}', recursive=True)
    return hits[0] if hits else None

QWEN8 = model_dir_opt('/qwen-3/', '/8b-base/')
QWEN14 = model_dir_opt('/qwen-3/', '/14b-base/')
AD_Q8 = adapter_dir('q8v5') or adapter_dir('q8v5_f0')
AD_Q14 = adapter_dir('q14v5') or adapter_dir('q14v5_f0')

print(f"QWEN8 Base: {QWEN8}, Adapter: {AD_Q8}", flush=True)
print(f"QWEN14 Base: {QWEN14}, Adapter: {AD_Q14}", flush=True)

res_q8 = evaluate_lora_backbone('Qwen3_8B_LoRA', QWEN8, AD_Q8, keep_layers=24, bs=16)
if res_q8:
    MODEL_PREDS['Qwen3_8B'], MODEL_TIMES['Qwen3_8B'] = res_q8

res_q14 = evaluate_lora_backbone('Qwen3_14B_LoRA', QWEN14, AD_Q14, keep_layers=27, bs=8)
if res_q14:
    MODEL_PREDS['Qwen3_14B'], MODEL_TIMES['Qwen3_14B'] = res_q14
"""))

# ----------------- CELL 7 -----------------
cells.append(md_cell("""### Model 6: Google Gemma-2-9B (DeepMind SentencePiece Orthogonal Backbone)"""))

cells.append(code_cell("""GEMMA9 = model_dir_opt('/gemma-2/', '/9b/')
gemma_hits = glob.glob('/kaggle/input/**/ntl-ts-gemma2-9b', recursive=True)
AD_GEMMA = gemma_hits[0] if gemma_hits else None

print(f"GEMMA9 Base: {GEMMA9}, Adapter: {AD_GEMMA}", flush=True)

res_gemma = evaluate_lora_backbone('Gemma2_9B_LoRA', GEMMA9, AD_GEMMA, keep_layers=24, bs=8)
if res_gemma:
    MODEL_PREDS['Gemma2_9B'], MODEL_TIMES['Gemma2_9B'] = res_gemma
"""))

# ----------------- CELL 8 -----------------
cells.append(md_cell("""### 📊 ARENA RESULTS: Solo Performance, Disagreement Matrix, & Greedy Ensemble"""))

cells.append(code_cell(r"""print("\n" + "="*80)
print("=== ARENA BENCHMARK: SOLO PERFORMANCE SUMMARY ===")
print("="*80)

summary_rows = []
for name, probs in MODEL_PREDS.items():
    acc = accuracy_score(y_val, probs.argmax(axis=1))
    loss = log_loss(y_val, probs)
    t = MODEL_TIMES.get(name, 0.0)
    summary_rows.append({"Model": name, "Accuracy": acc, "LogLoss": loss, "Time_s": t})

summary_df = pd.DataFrame(summary_rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
for idx, r in summary_df.iterrows():
    print(f"#{idx+1:02d} {r['Model']:22s} | Accuracy: {r['Accuracy']*100:.2f}% | LogLoss: {r['LogLoss']:.4f} | Time: {r['Time_s']:.1f}s")

# 2. Pairwise Disagreement Matrix (Diversity / Orthogonality)
print("\n" + "="*80)
print("=== PAIRWISE ERROR DISAGREEMENT MATRIX (HIGHER = MORE ORTHOGONAL) ===")
print("="*80)

model_names = list(MODEL_PREDS.keys())
disagree_mat = np.zeros((len(model_names), len(model_names)))

for i, m1 in enumerate(model_names):
    p1 = MODEL_PREDS[m1].argmax(axis=1)
    err1 = (p1 != y_val)
    for j, m2 in enumerate(model_names):
        p2 = MODEL_PREDS[m2].argmax(axis=1)
        err2 = (p2 != y_val)
        union_err = err1 | err2
        if union_err.sum() > 0:
            diff_err = (err1 != err2).sum()
            disagree_mat[i, j] = diff_err / union_err.sum()
        else:
            disagree_mat[i, j] = 0.0

dis_df = pd.DataFrame(disagree_mat, index=model_names, columns=model_names)
print(dis_df.round(3))

# 3. Greedy Forward Selection: Find the Apex Ensemble Combination
print("\n" + "="*80)
print("=== GREEDY FORWARD SELECTION: FINDING THE OPTIMAL ENSEMBLE ===")
print("="*80)

best_models = [summary_df.iloc[0]['Model']]
best_blend = MODEL_PREDS[best_models[0]].copy()
best_acc = summary_df.iloc[0]['Accuracy']
best_weights = {best_models[0]: 1.0}

print(f"Base Model: {best_models[0]} with Accuracy = {best_acc*100:.2f}%")

remaining = [m for m in model_names if m not in best_models]
step = 1
while remaining:
    best_candidate = None
    best_candidate_weight = 0.0
    best_candidate_acc = best_acc
    
    for cand in remaining:
        cand_probs = MODEL_PREDS[cand]
        for w in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            trial_blend = best_blend + w * cand_probs
            trial_acc = accuracy_score(y_val, trial_blend.argmax(axis=1))
            if trial_acc > best_candidate_acc:
                best_candidate_acc = trial_acc
                best_candidate = cand
                best_candidate_weight = w
                
    if best_candidate and best_candidate_acc > best_acc + 0.001:
        best_acc = best_candidate_acc
        best_models.append(best_candidate)
        best_weights[best_candidate] = best_candidate_weight
        best_blend = best_blend + best_candidate_weight * MODEL_PREDS[best_candidate]
        remaining.remove(best_candidate)
        print(f"Step {step}: Added {best_candidate:20s} (weight={best_candidate_weight:.1f}) -> New CV Accuracy = {best_acc*100:.2f}% (+{(best_acc - summary_df.iloc[0]['Accuracy'])*100:+.2f}%)")
        step += 1
    else:
        print("No further candidate model improves the ensemble accuracy. Stopping selection.")
        break

print("\n" + "="*80)
print("🏆 FINAL APEX ENSEMBLE RECIPE:")
print("="*80)
for m, w in best_weights.items():
    print(f"  • {m:22s}: Weight = {w:.2f}")
print(f"Total Ensemble Accuracy: {best_acc*100:.2f}% (Gain over best solo: +{(best_acc - summary_df.iloc[0]['Accuracy'])*100:.2f}%)")
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('model_arena/arena-benchmark.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("model_arena/arena-benchmark.ipynb generated successfully!")
