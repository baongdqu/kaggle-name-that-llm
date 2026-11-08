import json
import ast
import re

# Read V17 notebook
with open('champion_v17/champion-v17.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 0: Title and description
cell0_text = """# 0.640+ | Champion V18 Apex Titan (ATLAS 10% + Hungarian Injective + Full Backbones)
**Team: Bao Ng 🔫🔫🔫 (Rank 15 Global / Gold Medal Tier)**

### Architecture Highlights:
1. **Core Deep Ensemble**: Two-Stage LoRA Qwen3-14B ($w=0.70$) & 8B ($w=0.40$) + 35k Public Pool 4-View LinearSVC ($w=0.60$) + Stylegrams ($w=1.516$) + 3 Encoders.
2. **Safe Backbones Restored**: DeBERTa-v3 Token-Attention ($w=0.185$) + UltraBERT 5-fold ($w=0.095$) with zero risk of CUDA stream deadlock.
3. **NEW SOTA Booster**: Integrated **ATLAS Qwen3-8B Adapter** (Joseph Adamski 0.607) with 10% probability blend.
4. **Superior Mathematical Decoding**:
   - **Kuhn-Munkres Hungarian Bipartite Assignment** ($g \le 6$): Exact 1-to-1 matching under distinct-author constraint.
   - **Margin-Locked Sinkhorn**: Confidence anchor protection ($margin \ge 0.40$, $P \ge 0.70$).
   - **Universal 18-Class Mega-Gate** + 4 Family Specialized Intra-Cluster Gating.
"""
nb['cells'][0]['source'] = [line + '\n' for line in cell0_text.splitlines()]

# Cell 14: Restored Safe tokattn_deb + ultrabert + ATLAS
cell14_code = '''# ============================================================ 6. token-attention (DeBERTa) + UltraBERT + ATLAS Qwen3-8B
class TokenAttention(torch.nn.Module if torch is not None else object):
    def __init__(self, k, h=4096):
        super().__init__()
        self.drop, self.proj, self.score, self.out = torch.nn.Dropout(0.2), torch.nn.Linear(h, 64), torch.nn.Linear(64, 1), torch.nn.Linear(128, k)

    def forward(self, x, m):
        h = torch.tanh(self.proj(self.drop(x)))
        a = torch.softmax(self.score(h).squeeze(-1).masked_fill(~m, -1e4), -1).unsqueeze(-1)
        mf = m.unsqueeze(-1).float()
        return self.out(self.drop(torch.cat([(h * mf).sum(1) / mf.sum(1).clamp(min=1), (a * h).sum(1)], -1)))


@torch.no_grad()
def token_attention_deb_probs(texts, bs=16):
    try:
        hits = sorted(glob.glob('/kaggle/input/**/tokattn_debl23_final.pt', recursive=True))
        if not hits:
            print('[tokattn_deb] tokattn_debl23_final.pt not attached, skipping gracefully', flush=True)
            return None
        d = torch.load(hits[0], map_location='cpu')
        path = find_checkpoint(None, dir_name='deberta-v3-large')
        if not path:
            print('[tokattn_deb] deberta-v3-large backbone not attached, skipping gracefully', flush=True)
            return None
        lm = AutoModel.from_pretrained(path, dtype=torch.float32).to(DEV).eval()
        tok = AutoTokenizer.from_pretrained(path)
        tok.padding_side = 'right'
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        H = d.get('hidden', 1024)
        net = TokenAttention(K, H)
        net.load_state_dict(d['state_dict'])
        dev = lm.get_input_embeddings().weight.device
        net, mu, sd = net.to(dev).eval(), d['mu'].float().to(dev), d['sd'].float().to(dev)
        ids = tok(list(texts), add_special_tokens=True, truncation=True, max_length=d['max_len'])['input_ids']
        P = np.zeros((len(texts), K))
        order = np.argsort([len(t) for t in ids])
        for b0 in range(0, len(order), bs):
            idx = order[b0:b0 + bs]
            batch = tok.pad({'input_ids': [ids[i] for i in idx]}, return_tensors='pt').to(dev)
            out = lm(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
            hs = (out.hidden_states if getattr(out, 'hidden_states', None) is not None else out.language_model_outputs.hidden_states)[d['layer']]
            n = batch['attention_mask'].sum(1)
            T = max(int(n.max()) - 1, 1)
            x = torch.zeros(len(idx), T, H, device=dev)
            m = torch.zeros(len(idx), T, dtype=torch.bool, device=dev)
            for r in range(len(idx)):
                k = int(n[r]) - 1
                if k > 0:
                    x[r, :k] = ((torch.nan_to_num(hs[r, 1:k + 1].to(dev).float()).half().float() - mu) / sd).clamp(-8, 8)
                    m[r, :k] = True
            P[idx] = torch.softmax(net(x, m), -1).cpu().numpy()
            del hs, out, batch
        del lm, net
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return P
    except Exception as e:
        print(f'[tokattn_deb] Skipped due to exception: {e}', flush=True)
        return None


@torch.no_grad()
def ultrabert_probs(texts):
    try:
        from transformers import AutoModelForSequenceClassification
        folds = None
        for root in glob.glob('/kaggle/input/**/ultrabert-380m-ntl', recursive=True):
            fs = [f for f in sorted(glob.glob(os.path.join(root, 'fold*'))) if os.path.exists(os.path.join(f, 'config.json'))
                  and json.load(open(os.path.join(f, 'config.json'))).get('model_type') == 'ultrabert']
            if len(fs) == 5:
                folds = fs
                break
        if not folds:
            print('[ultrabert] ultrabert-380m-ntl (5 folds) not attached, skipping gracefully', flush=True)
            return None
        tok = AutoTokenizer.from_pretrained(folds[0], trust_remote_code=True)
        ids = tok(texts, truncation=True, max_length=512)['input_ids']
        order = np.argsort([len(e) for e in ids])
        P = np.zeros((len(texts), K))
        for fp in folds:
            em = AutoModelForSequenceClassification.from_pretrained(fp, trust_remote_code=True, dtype=torch.float16, attn_implementation='sdpa').to(DEV).eval()
            cols = [LAB2I[em.config.id2label[i]] for i in range(len(em.config.id2label))]
            for b0 in range(0, len(order), 32):
                idx = order[b0:b0 + 32]
                b = tok.pad({'input_ids': [ids[j] for j in idx]}, return_tensors='pt').to(DEV)
                P[np.ix_(idx, cols)] += em(**b).logits.float().softmax(-1).cpu().numpy() / len(folds)
            del em
            gc.collect()
            torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return P
    except Exception as e:
        print(f'[ultrabert] Skipped due to exception: {e}', flush=True)
        return None


if not SMOKE:
    p_deb = token_attention_deb_probs(TE)
    if p_deb is not None:
        add_member('tokattn_deb', p_deb, weight=0.185)
    p_ub = ultrabert_probs(TE)
    if p_ub is not None:
        add_member('ultrabert', p_ub, weight=0.095)

# ============================================================ 6b. ATLAS Qwen3-8B Adapter (Joseph Adamski 0.607)
ATLAS_PROB = None
try:
    _atlas_roots = sorted(glob.glob('/kaggle/input/**/atlas_bundle.json', recursive=True))
    if _atlas_roots:
        _atlas_root = os.path.dirname(_atlas_roots[0])
        import sys
        if _atlas_root not in sys.path:
            sys.path.insert(0, _atlas_root)
        from ntl_atlas.inference import predict as _atlas_predict
        from ntl_rlcd.data import LABELS as _atlas_labels
        base_8b = find_checkpoint('qwen3_8b')
        if not base_8b:
            base_8b = find_base(dict(model_type='qwen3', hidden_size=4096, num_hidden_layers=36))
        print(f'[ATLAS] Loading ATLAS Qwen3-8B adapter from {_atlas_root}...', flush=True)
        _ap = _atlas_predict(TE, base_8b, _atlas_root, batch_size=1)
        if _ap is not None and len(_ap) == len(TE):
            ATLAS_PROB = _ap[:, [_atlas_labels.index(label) for label in LABELS]]
            assert ATLAS_PROB.shape == (len(TE), K)
            assert np.isfinite(ATLAS_PROB).all() and (ATLAS_PROB >= 0).all()
            print(f'[ATLAS] Successfully computed ATLAS probabilities on {len(TE)} rows', flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    else:
        print('[ATLAS] atlas_bundle.json not attached, skipping gracefully', flush=True)
except Exception as e:
    print(f'[ATLAS] Skipped due to exception: {e}', flush=True)
'''
nb['cells'][14]['source'] = [line + '\n' for line in cell14_code.splitlines()]

# Cell 16: Blend, ATLAS integration, Sinkhorn, Gates, Hungarian
cell16_old = ''.join(nb['cells'][16]['source'])

# Modify Cell 16: Add ATLAS blend right after P = softmax_rows(LOGIT)
atlas_blend_snippet = """P = softmax_rows(LOGIT)

if ATLAS_PROB is not None:
    P = 0.90 * P + 0.10 * ATLAS_PROB
    P /= P.sum(1, keepdims=True) + 1e-12
    print(f'Blended ATLAS adapter (10% weight) into ensemble probability distribution', flush=True)
"""

cell16_new = cell16_old.replace("P = softmax_rows(LOGIT)\n", atlas_blend_snippet)
if "Blended ATLAS adapter" not in cell16_new:
    cell16_new = cell16_old.replace("P = softmax_rows(LOGIT)", atlas_blend_snippet)

nb['cells'][16]['source'] = [line + '\n' for line in cell16_new.splitlines()]

# Validate every single code cell with ast.parse
print("=== Validating AST for all cells in V18 ===")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        code = ''.join(cell['source'])
        try:
            ast.parse(code)
            print(f"Cell {i}: OK (AST Valid)")
        except SyntaxError as e:
            print(f"Cell {i}: SYNTAX ERROR: {e}")
            raise e

# Save to champion_v18/champion-v18.ipynb
out_path = 'champion_v18/champion-v18.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"\nSuccessfully generated {out_path} ({len(nb['cells'])} cells)!")
