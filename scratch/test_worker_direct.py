
import json, math, os, random, sys, time
import importlib.util as _iu
_fs = _iu.find_spec
_iu.find_spec = lambda name, *a, **k: None if name.split('.')[0] == 'torchao' else _fs(name, *a, **k)
import numpy as np, pandas as pd, torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
import peft.import_utils as _pi
_pi.is_torchao_available = lambda *a, **k: False
from safetensors.torch import save_file

cfg = json.loads(sys.argv[1]); jobs = sys.argv[2].split(',')
train = pd.read_csv(cfg['train_csv'])
LABELS = sorted(train.model.unique()); y = train.model.map({l: i for i, l in enumerate(LABELS)}).values
pid = train.prompt_id.values
prompts = np.array(sorted(train.prompt_id.unique()))
rng0 = np.random.RandomState(cfg['fold_seed']); rng0.shuffle(prompts)
fold = train.prompt_id.map({p: i % 5 for i, p in enumerate(prompts)}).values
tok = AutoTokenizer.from_pretrained(cfg['model_dir'])
tok.padding_side = 'right'
if tok.pad_token is None: tok.pad_token = tok.eos_token
tokens = [ids + [tok.eos_token_id] for ids in
          tok(train.text.tolist(), add_special_tokens=False, truncation=True, max_length=cfg['max_len'])['input_ids']]
L = cfg['layers']; dev = torch.device('cuda:0')

# Sibling families definition
SIBLING_FAMILIES = {
    "gemini_flash": ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"],
    "glm_53":       ["glm-5.3", "glm-5.3-flash"],
    "gpt_56":       ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"],
}

# Build asymmetric penalty matrix W
W_PENALTY = np.ones((len(LABELS), len(LABELS)), dtype=np.float32)
for f_name, members in SIBLING_FAMILIES.items():
    idxs = [LABELS.index(m) for m in members if m in LABELS]
    for i in idxs:
        for j in idxs:
            if i != j:
                W_PENALTY[i, j] = 3.0
W_DEV = torch.as_tensor(W_PENALTY, device=dev)

def weighted_multi_similarity_loss(z_norm, targets, W_dev, alpha=2.0, beta=40.0, base=0.5):
    # z_norm: (B, D) unit normalized features
    sim = torch.matmul(z_norm, z_norm.T)  # (B, B)
    B = z_norm.shape[0]
    loss = 0.0
    valid_count = 0
    for i in range(B):
        pos_mask = (targets == targets[i])
        pos_mask[i] = False
        neg_mask = (targets != targets[i])
        
        if not pos_mask.any() or not neg_mask.any():
            continue
            
        pos_sim = sim[i, pos_mask]
        neg_sim = sim[i, neg_mask]
        neg_w = W_dev[targets[i], targets[neg_mask]]
        
        pos_term = (1.0 / alpha) * torch.log(1.0 + torch.sum(torch.exp(-alpha * (pos_sim - base))))
        neg_term = (1.0 / beta) * torch.log(1.0 + torch.sum(neg_w * torch.exp(beta * (neg_sim - base))))
        loss += (pos_term + neg_term)
        valid_count += 1
    return loss / max(valid_count, 1)

def sibling_triplets(rows, rng):
    # Standard same-prompt triplets preserved
    out = []
    for p in np.unique(pid[rows]):
        m = rng.permutation(rows[pid[rows] == p])
        for s in range(0, len(m), 3):
            b = list(m[s:s + 3])
            while len(b) < 3:
                b.append(int(rng.choice([x for x in m if x not in b])))
            out.append(b)
            
    # Add Sibling-Mining microbatches (cross-prompt sibling contrasts)
    for f_name, members in SIBLING_FAMILIES.items():
        m_idxs = [LABELS.index(m) for m in members if m in LABELS]
        f_rows = rows[np.isin(y[rows], m_idxs)]
        if len(f_rows) >= 3:
            f_perm = rng.permutation(f_rows)
            for s in range(0, len(f_perm) - 2, 3):
                out.append(list(f_perm[s:s + 3]))
                
    out = [out[i] for i in rng.permutation(len(out))]
    while len(out) % cfg['accum']: out.append(out[rng.randint(len(out))])
    return out

for job in jobs:
    t0 = time.time()
    outdir = os.path.join('lora_out', f'job_{job}'); os.makedirs(outdir, exist_ok=True)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    rng = np.random.RandomState(cfg['seed'] + (99 if job == 'final' else int(job)))
    fit = np.arange(len(train)) if job == 'final' else np.where(fold != int(job))[0]
    valid = np.arange(3) if job == 'final' else np.where(fold == int(job))[0]
    mc = AutoConfig.from_pretrained(cfg['model_dir'])
    mc.num_hidden_layers, mc.use_cache = cfg['keep'], False
    if getattr(mc, 'layer_types', None) is not None: mc.layer_types = mc.layer_types[:cfg['keep']]
    base = AutoModel.from_pretrained(cfg['model_dir'], config=mc, dtype=torch.float16).to(dev)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(r=cfg['r'], lora_alpha=cfg['alpha'], lora_dropout=cfg['ldrop'], bias='none',
                           target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    for p in model.parameters():
        if p.requires_grad: p.data = p.data.float()
    norms = nn.ModuleList([nn.LayerNorm(4096, eps=1e-5) for _ in L]).to(dev).float()
    plain = nn.Linear(4096 * len(L), len(LABELS)).to(dev).float()
    centered = nn.Linear(4096 * len(L), len(LABELS)).to(dev).float()
    drop = nn.Dropout(cfg['ldrop'])
    ad = [p for p in model.parameters() if p.requires_grad]
    hd = list(norms.parameters()) + list(plain.parameters()) + list(centered.parameters())
    opt = torch.optim.AdamW([{'params': ad, 'lr': cfg['lr_adapter']}, {'params': hd, 'lr': cfg['lr_head']}],
                            weight_decay=cfg['wd'])
    plan = [sibling_triplets(fit, rng) for _ in range(cfg['epochs'])]
    total = sum(len(b) for b in plan) // cfg['accum']
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total - int(total * cfg['warmup'])))
    scaler = torch.amp.GradScaler('cuda', init_scale=256)

    def rep(indices):
        batch = tok.pad({'input_ids': [tokens[i] for i in indices]}, padding=True, return_tensors='pt').to(dev)
        hs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                   output_hidden_states=True).hidden_states
        m = batch['attention_mask'].unsqueeze(-1).float(); m[:, 0] = 0
        with torch.autocast('cuda', enabled=False):
            return torch.cat([nm((hs[l].float() * m).sum(1) / m.sum(1).clamp(min=1)) for l, nm in zip(L, norms)], -1)

    model.train(); norms.train(); step = 0; skipped = 0; losses = []
    for ep, batches in enumerate(plan):
        # Linear warm-up of contrastive penalty term
        lambda_ms = min(0.20, 0.05 + 0.15 * (ep / max(1, cfg['epochs'] - 1)))
        for i, b in enumerate(batches):
            z = rep(b)
            yy = torch.as_tensor(y[b], device=dev)
            
            # CE Losses
            loss_ce_plain = nn.functional.cross_entropy(plain(drop(z)), yy)
            loss_ce_cent = nn.functional.cross_entropy(centered(drop(z - z.mean(0, keepdim=True))), yy)
            
            # Sibling-Aware Contrastive Loss
            z_norm = nn.functional.normalize(z, p=2, dim=-1)
            loss_ms = weighted_multi_similarity_loss(z_norm, yy, W_DEV)
            
            loss = loss_ce_plain + loss_ce_cent + lambda_ms * loss_ms
            scaler.scale(loss / cfg['accum']).backward(); losses.append(float(loss.detach()))
            if (i + 1) % cfg['accum'] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(ad + hd, 1.0)
                before = scaler.get_scale(); scaler.step(opt); scaler.update()
                if scaler.get_scale() < before: skipped += 1
                opt.zero_grad(set_to_none=True); sch.step(); step += 1
                if step % 30 == 0:
                    print(f'job {job} ep {ep} step {step}/{total} loss {np.mean(losses[-180:]):.4f} skipped {skipped} '
                          f'({time.time() - t0:.0f}s)', flush=True)
    model.eval(); norms.eval()
    Z = np.zeros((len(valid), 4096 * len(L)), np.float32)
    order = np.argsort([len(tokens[i]) for i in valid], kind='stable')
    with torch.inference_mode():
        for s in range(0, len(order), 6):
            pos = order[s:s + 6]
            Z[pos] = rep(valid[pos]).cpu().numpy()
    assert np.isfinite(Z).all()
    W = {n: t.weight.detach().cpu().numpy() for n, t in (('plain', plain), ('center', centered))}
    B = {n: t.bias.detach().cpu().numpy() for n, t in (('plain', plain), ('center', centered))}
    np.savez_compressed(os.path.join(outdir, 'readouts.npz'), classes=np.array(LABELS), pool_layers=np.array(L),
                        plain_coef=W['plain'], plain_intercept=B['plain'], center_coef=W['center'], center_intercept=B['center'])
    np.savez_compressed(os.path.join(outdir, 'valid_features.npz'), ids=train.id.values[valid], features=Z.astype(np.float16))
    lg = Z @ W['plain'].T + B['plain']
    acc = float((lg.argmax(1) == y[valid]).mean())
    if job == 'final':
        model.save_pretrained(os.path.join(outdir, 'adapter'), safe_serialization=True)
        save_file({**{f'norms.{k}': v.detach().cpu().contiguous() for k, v in norms.state_dict().items()}},
                  os.path.join(outdir, 'norms.safetensors'))
        tok.save_pretrained(os.path.join(outdir, 'tokenizer'))
    json.dump({'job': job, 'valid_acc': acc, 'steps': step, 'skipped': skipped, 'secs': time.time() - t0},
              open(os.path.join(outdir, 'COMPLETE.json'), 'w'))
    print(f'job {job} done: valid acc {acc:.4f} steps {step} skipped {skipped} ({time.time() - t0:.0f}s)', flush=True)
    del model, base, opt, norms, plain, centered; torch.cuda.empty_cache()
