"""Name That LLM inference: two-stage LoRA24 ensemble, test-prompt structure, ATLAS, class balancing."""
import os, sys, time, gc, argparse, json, re
import numpy as np, pandas as pd, torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from transformers import AutoTokenizer, AutoModel, AutoConfig
from safetensors.torch import load_file
import gauss

LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3', 'glm-5.3-flash',
          'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']
K = len(LABELS)
L2I = {l: i for i, l in enumerate(LABELS)}

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--members', required=True)
p.add_argument('--pool', required=True)
p.add_argument('--svc_w', type=float, default=1.5)
p.add_argument('--twin_w', type=float, default=1.0)
p.add_argument('--cluster_thr', type=float, default=0.15)
p.add_argument('--comp_path', default='')
p.add_argument('--comp_w', type=float, default=0.15)
p.add_argument('--atlas_dir', default='')
p.add_argument('--atlas_base', default='')
p.add_argument('--atlas_w', type=float, default=0.10)
p.add_argument('--atlas_margin', type=float, default=0.3)
p.add_argument('--atlas_bs', type=int, default=4)
p.add_argument('--atlas_chunk', type=int, default=3)
p.add_argument('--budget', type=float, default=8.3 * 3600)
p.add_argument('--out', default='submission.parquet')
a = p.parse_args()

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
T0 = time.time()
def log(*x): print(f'[{time.time() - T0:7.0f}s]', *x, flush=True)

def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))

def topic_correction(core, svc, groups, alpha=0.62, weight=1.0):
    """Remove prompt topic bias in inferred dense groups; no class assignment.
    Centers SVC evidence within prompt groups (8 <= size <= 18) to isolate pure authorial style."""
    core = np.asarray(core, dtype=np.float64); groups = np.asarray(groups)
    if alpha == 0 or svc is None or weight == 0:
        return core.copy()
    scores = np.asarray(svc, dtype=np.float64)
    if scores.shape != core.shape or len(groups) != len(core):
        return core.copy()
    centered = scores.copy()
    _, inverse, count = np.unique(groups, return_inverse=True, return_counts=True)
    eligible = (count[inverse] >= 8) & (count[inverse] <= 18)
    for g in np.unique(groups[eligible]):
        ix = (groups == g)
        centered[ix] -= scores[ix].mean(0, keepdims=True)
    delta = log_softmax(centered) - log_softmax(scores)
    delta[~eligible] = 0
    ordered = np.sort(core, axis=1)
    gate = np.clip((2.0 - (ordered[:, -1] - ordered[:, -2])) / 1.6, 0.15, 1.0)
    gate[~eligible] = 0
    return core + float(alpha) * float(weight) * delta * gate[:, None]


train = pd.read_csv(os.path.join(a.data, 'train.csv'))
test = pd.read_csv(os.path.join(a.data, 'test.csv'))
train['text'] = train.text.fillna(''); test['text'] = test.text.fillna('')
y = train.model.map(L2I).values
TINY = len(test) < 50
log('train', len(train), 'test', len(test))


# ---------------------------------------------------------------- test prompt clusters
def capped_agglo(S, cap=18, thr=0.15):
    """Average-linkage clustering that never grows a cluster past `cap` (one passage per model per prompt)."""
    n = len(S); A = S.astype(np.float32).copy(); np.fill_diagonal(A, -np.inf)
    size = np.ones(n, np.int32); alive = np.ones(n, bool); lab = np.arange(n)
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        M = np.where(ok, A, -np.inf); i, j = divmod(int(np.argmax(M)), n)
        if not np.isfinite(M[i, j]) or M[i, j] < thr:
            break
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new; A[:, i] = new; A[i, i] = -np.inf; A[j] = -np.inf; A[:, j] = -np.inf
        size[i] += size[j]; alive[j] = False; lab[lab == j] = i
    return lab

try:
    Xc = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05).fit_transform(test.text.values)
    groups = capped_agglo((Xc @ Xc.T).toarray(), 18, a.cluster_thr)
except ValueError:
    groups = np.arange(len(test))
groups = np.unique(groups, return_inverse=True)[1]
gsize = np.bincount(groups)[groups]
log('test clusters', groups.max() + 1, '| passages in clusters of 2-18:', int(((gsize >= 2) & (gsize <= 18)).sum()))

LP = {}


# ---------------------------------------------------------------- 4-view n-gram SVM
FUNCTION_WORDS = set("""a an the and or but if while although because as so than that which who whom whose this these those it
its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for
of by over under between through into about is are was were be been being have has had do does did can could will would shall
should may might must not no nor all any some each every both either neither other another such only also just very too more
most less least here there when where why how what whether then now still yet even much many few little""".split())

def shape_view(t): return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)
def function_view(t):
    return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT" for w in re.findall(r"\w+|[^\w\s]", t))

pool = pd.read_parquet(a.pool); pool = pool[pool.model.isin(LABELS)]
fit_text = list(train.text.values) + list(pool.text.astype(str).values)
fit_y = np.r_[y, pool.model.map(L2I).values]
A_, B_ = [], []
for view, lo, hi, max_feat, w in [('char', 3, 5, 250000, 1.0), ('word', 1, 3, 180000, 0.7),
                                   ('shape', 2, 6, 60000, 0.5), ('function', 1, 3, 50000, 0.5)]:
    prep = {'shape': shape_view, 'function': function_view}.get(view, lambda s: s)
    analyzer = {'shape': 'char', 'function': 'word'}.get(view, view)
    kw = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=max_feat, dtype=np.float32)
    if analyzer == 'word':
        kw['token_pattern'] = r"(?u)\b\w+\b|[^\w\s]"
    vec = TfidfVectorizer(**kw)
    A_.append(vec.fit_transform([prep(t) for t in fit_text]) * w)
    B_.append(vec.transform([prep(t) for t in test.text.values]) * w)
svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight='balanced').fit(hstack(A_, format='csr'), fit_y)
LP['svc4'] = log_softmax(svc.decision_function(hstack(B_, format='csr'))) * a.svc_w
del A_, B_, svc; gc.collect(); log('svc4 done')


# ---------------------------------------------------------------- two-stage LoRA24 members
def batches(enc, bs):
    order = np.argsort([len(x) for x in enc])
    for i in range(0, len(order), bs):
        b = order[i:i + bs]; L = max(len(enc[j]) for j in b)
        ids = torch.zeros((len(b), L), dtype=torch.long); att = torch.zeros((len(b), L), dtype=torch.long)
        for r, j in enumerate(b):
            ids[r, :len(enc[j])] = torch.tensor(enc[j]); att[r, :len(enc[j])] = 1
        yield b, ids, att

@torch.no_grad()
def lora24(spec, texts, groups=None, train_texts=None):
    """Truncated base loaded once; each adapter is merged into the weights (W += BA * alpha/r), read, then unmerged.
    Features: masked mean-pool (first token dropped) of several layers, each through its own LayerNorm.
    Returns plain-head and centred-head log-probs averaged over adapters, plus (test, train) features of the last adapter."""
    tok = AutoTokenizer.from_pretrained(spec['path'])
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    cfg = AutoConfig.from_pretrained(spec['path']); tc = getattr(cfg, 'text_config', cfg)
    tc.num_hidden_layers = spec['keep']; tc.use_cache = False
    if getattr(tc, 'layer_types', None) is not None:
        tc.layer_types = tc.layer_types[:spec['keep']]
    m = AutoModel.from_pretrained(spec['path'], config=cfg, dtype=torch.float16, device_map='auto').eval()
    if spec.get('no_final_norm'):
        m.norm = torch.nn.Identity()
    mods = dict(m.named_modules()); dev = next(m.parameters()).device; layers = spec['layers']
    eos = spec.get('eos_id')   # checkpoints trained with an end token appended need the same id here
    def tokenize(tx):
        ids = tok(list(tx), truncation=True, max_length=spec['max_len'] - (1 if eos is not None else 0))['input_ids']
        return [x + [eos] for x in ids] if eos is not None else ids
    enc = tokenize(texts)
    enc_tr = tokenize(train_texts) if train_texts is not None else None

    def featurize(enc_, norms, hid):
        Z = np.zeros((len(enc_), hid * len(layers)), np.float32)
        for b, ids, att in batches(enc_, spec['bs']):
            ids[att == 0] = pad
            hs = m(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True, use_cache=False).hidden_states
            if not all(torch.isfinite(hs[l]).all() for l in layers):   # fp16 overflow: redo the batch in bf16
                m.to(torch.bfloat16); hs = m(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True, use_cache=False).hidden_states; m.to(torch.float16)
            mask = att.unsqueeze(-1).float(); mask[:, 0] = 0
            Z[b] = torch.cat([nm((hs[l].float().cpu() * mask).sum(1) / mask.sum(1)) for l, nm in zip(layers, norms)], -1).numpy()
        return Z

    plain, cent, feats = 0.0, 0.0, None
    for ai, ad in enumerate(spec['adapters']):
        adp = os.path.join(ad, 'adapter') if os.path.exists(os.path.join(ad, 'adapter', 'adapter_config.json')) else ad
        acfg = json.load(open(os.path.join(adp, 'adapter_config.json'))); scale = acfg['lora_alpha'] / acfg['r']
        sd = load_file(os.path.join(adp, 'adapter_model.safetensors')); merged = []
        for k, A in sd.items():
            if '.lora_A.' not in k:
                continue
            name = k.replace('base_model.model.', '', 1).split('.lora_A.')[0]
            if name not in mods and 'model.' + name in mods:
                name = 'model.' + name
            W = mods[name].weight; B = sd[k.replace('.lora_A.', '.lora_B.')]
            W += ((B.to(W.device, torch.float32) @ A.to(W.device, torch.float32)) * scale).to(W.dtype)
            merged.append((W, B, A))
        H = torch.load(os.path.join(ad, 'heads.pt'), map_location='cpu') if os.path.exists(os.path.join(ad, 'heads.pt')) \
            else dict(load_file(os.path.join(ad, 'heads.safetensors')))
        if os.path.exists(os.path.join(ad, 'meta.json')):
            order = [json.load(open(os.path.join(ad, 'meta.json')))['labels'].index(l) for l in LABELS]
            for h in ('plain', 'centered'):
                H[f'{h}.weight'], H[f'{h}.bias'] = H[f'{h}.weight'][order], H[f'{h}.bias'][order]
        hid = H['norms.0.weight'].shape[0]
        norms = [torch.nn.LayerNorm(hid) for _ in layers]
        for i, nm in enumerate(norms):
            nm.load_state_dict({'weight': H[f'norms.{i}.weight'], 'bias': H[f'norms.{i}.bias']})
        Z = featurize(enc, norms, hid)
        if enc_tr is not None and ai == len(spec['adapters']) - 1:
            feats = (Z.copy(), featurize(enc_tr, norms, hid))
        for W, B, A in merged:
            W -= ((B.to(W.device, torch.float32) @ A.to(W.device, torch.float32)) * scale).to(W.dtype)
        del merged, sd; gc.collect(); torch.cuda.empty_cache()
        head = lambda h, X: log_softmax(X @ H[f'{h}.weight'].float().numpy().T + H[f'{h}.bias'].float().numpy())
        plain = plain + head('plain', Z) / len(spec['adapters'])
        if groups is not None:
            Zc = Z.copy()
            for g in np.unique(groups):
                Zc[groups == g] -= Zc[groups == g].mean(0)
            cent = cent + head('centered', Zc) / len(spec['adapters'])
        log(f'  {spec["name"]}: {os.path.basename(ad)}')
    del m, mods; gc.collect(); torch.cuda.empty_cache()
    return plain, (cent if groups is not None else None), feats

@torch.no_grad()
def encoder_member(ckpt, texts):
    """Fully fine-tuned encoder (DeBERTa / ModernBERT): fp32 weights, mean of the last layer, linear head."""
    meta = json.load(open(os.path.join(ckpt, 'meta.json'))); perm = [meta['labels'].index(l) for l in LABELS]
    head = load_file(os.path.join(ckpt, 'head.safetensors')); W = head['weight'].float().numpy()[perm]; b = head['bias'].float().numpy()[perm]
    etok = AutoTokenizer.from_pretrained(os.path.join(ckpt, 'encoder'))
    enc = AutoModel.from_pretrained(os.path.join(ckpt, 'encoder'), dtype=torch.float32).to('cuda:0').eval()
    def feats(tx):
        ids = etok(list(tx), truncation=True, max_length=meta['max_len'])['input_ids']
        order = np.argsort([len(x) for x in ids], kind='stable'); Z = np.zeros((len(ids), enc.config.hidden_size), np.float32)
        for s in range(0, len(order), 32):
            idx = order[s:s + 32]; bt = etok.pad({'input_ids': [ids[i] for i in idx]}, return_tensors='pt').to('cuda:0')
            mk = bt['attention_mask'].unsqueeze(-1).float()
            Z[idx] = ((enc(**bt).last_hidden_state * mk).sum(1) / mk.sum(1).clamp(min=1)).cpu().numpy()
        return Z
    if TINY:
        acc = float(((feats(train.text.values[:40]) @ W.T + b).argmax(1) == y[:40]).mean()); log(f'encoder in-sample check {acc:.2f}'); assert acc > 0.5
    out = log_softmax(feats(texts) @ W.T + b)
    del enc; gc.collect(); torch.cuda.empty_cache()
    return out

GAUSS_CFG = dict(red='wpca', d=128, se_shrink='lw', sp_shrink='lw', lam_floor=1e-4, n_exact=3, inj=True, iters=5, repel=True, sp_scale=1.0)
grouped = (gsize >= 2) & (gsize <= 18)
for spec in json.loads(a.members):
    name, w = spec['name'], spec['w']
    if spec.get('type') == 'enc':
        LP[name] = encoder_member(spec['ckpt'], test.text.values) * w; log(name, 'done'); continue
    if TINY:   # interactive 3-row run: the heads must reproduce their fit on training rows
        acc = float((lora24(spec, train.text.values[:40])[0].argmax(1) == y[:40]).mean())
        log(f'{name} in-sample check {acc:.2f}'); assert acc > 0.5, name
    use_gauss = spec.get('gauss', 0) > 0
    plain, cent, feats = lora24(spec, test.text.values, groups, train.text.values if use_gauss else None)
    LP[name] = plain * w
    LP[name + '_twin'] = np.where((gsize >= 2)[:, None], cent, 0.0) * a.twin_w * w * spec['twin']
    if use_gauss:   # shared-prompt Gaussian group head on the full-data adapter's features
        Zte, Ztr = feats
        prm = gauss.fit(Ztr, y, train.prompt_id.values, GAUSS_CFG); prm['T1'] = prm['Tg'] = 3.0
        assert np.allclose(np.exp(gauss.scores(Ztr[[0, 1]], np.zeros(2, dtype=int), prm)).sum(1), 1)
        G = np.zeros((len(test), K))
        if grouped.any():
            G[grouped] = gauss.scores(Zte[grouped], np.unique(groups[grouped], return_inverse=True)[1], prm)
        LP[name + '_gauss'] = G * spec['gauss']
    gc.collect(); torch.cuda.empty_cache(); log(name, 'done')


# ---------------------------------------------------------------- blend
n_terms = sum(np.abs(v).max() > 0 for v in LP.values())
z = sum(LP.values()) / max(1, n_terms)
P = np.exp(z - z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)

if a.comp_w > 0 and a.comp_path and os.path.exists(a.comp_path):   # complementary frozen-feature stack, run in its own process
    meta = json.load(open(a.comp_path.replace('comp_logit.npy', 'comp_meta.json')))
    assert meta['ids'] == test['id'].astype(str).tolist()
    cl = np.load(a.comp_path)[:, [meta['labels'].index(l) for l in LABELS]]
    C = np.exp(cl - cl.max(1, keepdims=True)); C /= C.sum(1, keepdims=True)
    P = (1 - a.comp_w) * P + a.comp_w * C; P /= P.sum(1, keepdims=True)
    log(f"complementary stack mixed at {a.comp_w}")
    # Apply Prompt Topic-Bias De-noising on LinearSVC stylometrics
    P_corr = topic_correction(P, C, groups, alpha=0.62, weight=a.comp_w)
    P = np.clip(P_corr, 1e-12, None); P /= P.sum(1, keepdims=True)
    log("topic_correction applied on dense prompt groups (alpha=0.62)")
else:
    log('complementary stack not available, skipped')


@torch.no_grad()
def atlas(bundle, rows):
    """Source-conditioned likelihood (ATLAS, arXiv 2610.19583) from a public bundle; shortest passages first,
    stops early if the token-proportional projection would overrun the time budget."""
    sys.path.insert(0, bundle)
    from ntl_atlas.model import load_model
    from ntl_rlcd.data import LABELS as ALAB
    cfg = json.load(open(os.path.join(bundle, 'atlas_bundle.json')))['training_config']
    model, tok = load_model(a.atlas_base, ALAB, cfg, device='auto')
    try:
        model.load_trainable(load_file(os.path.join(bundle, 'adapters.safetensors'))); model.eval()
        enc = tok(test.text.values[rows].tolist(), truncation=True, max_length=cfg['max_length'])['input_ids']
        order = np.argsort([len(t) for t in enc], kind='stable'); out = np.zeros((len(rows), K), np.float32)
        ctok = np.cumsum([len(enc[i]) for i in order]).astype(np.float64); t1 = time.time()
        for s in range(0, len(order), a.atlas_bs):
            idx = order[s:s + a.atlas_bs]
            pack = tok.pad({'input_ids': [enc[i] for i in idx]}, padding=True, return_tensors='pt').to(model.device)
            out[idx] = model.score(pack['input_ids'], pack['attention_mask'], source_chunk=a.atlas_chunk)[0].float().cpu().numpy()
            done = s + len(idx)
            if s % (25 * a.atlas_bs) == 0:
                eta = (time.time() - T0) + (time.time() - t1) / ctok[done - 1] * (ctok[-1] - ctok[done - 1]) * 1.1
                log(f'  atlas {done}/{len(order)}, projected finish {eta:.0f}s')
                if done > 50 and eta > a.budget:
                    raise RuntimeError('time budget')
        assert np.isfinite(out).all() and np.allclose(out.sum(1), 1, atol=1e-4)
        return out[:, [ALAB.index(l) for l in LABELS]]
    finally:
        del model; gc.collect(); torch.cuda.empty_cache()

if a.atlas_dir and a.atlas_w > 0:   # a 10% mix can only flip rows where the ensemble is close, so score only those
    top2 = np.sort(P, 1)[:, -2:]
    rows = np.arange(len(test)) if TINY else np.where(top2[:, 1] - top2[:, 0] < a.atlas_margin)[0]
    log(f'atlas on {len(rows)}/{len(test)} passages')
    try:
        P[rows] = (1 - a.atlas_w) * P[rows] + a.atlas_w * atlas(a.atlas_dir, rows)
        P /= P.sum(1, keepdims=True); log(f'atlas mixed at {a.atlas_w}')
    except Exception as e:
        log('atlas skipped:', repr(e))


# ---------------------------------------------------------------- class balancing and constraints
# Sibling Micro-Gates (V44 Penta-Family Synergy Disambiguation)
idx_g37 = LABELS.index('gemini-3.7-flash')
idx_g38 = LABELS.index('gemini-3.8-flash')
idx_glm = LABELS.index('glm-5.3')
idx_glmf = LABELS.index('glm-5.3-flash')
idx_cl_op5 = LABELS.index('claude-opus-5')
idx_cl_op46 = LABELS.index('claude-opus-4-6')
idx_gpt55 = LABELS.index('gpt-5.5')
idx_astra = LABELS.index('gpt-6-astra')
idx_terra = LABELS.index('gpt-5.6-terra')
idx_luna = LABELS.index('gpt-5.6-luna')

g38_boosts = 0
glm_boosts = 0
claude_boosts = 0
gpt_boosts = 0
astra_boosts = 0
kimi_boosts = 0
qwen_boosts = 0
sol_boosts = 0

for i, txt in enumerate(test.text.values):
    # 1. Gemini reasoning prefix gate:
    if any(m in txt for m in ['First,', 'Second,', 'Step 1:', '1. ', 'In conclusion,']):
        if P[i, idx_g38] > 0.04:
            P[i, idx_g38] *= 1.12
            P[i, idx_g37] *= 0.90
            g38_boosts += 1
            
    # 2. GLM Chinese Unicode marks gate:
    if any(m in txt for m in ['，', '。', '「', '」', '、', '《', '》']):
        if P[i, idx_glm] > 0.04:
            P[i, idx_glm] *= 1.15
            P[i, idx_glmf] *= 0.88
            glm_boosts += 1

    # 3. Claude spaced em-dash syntactic gate:
    if ' \u2014 ' in txt:
        if P[i, idx_cl_op5] > 0.04 or P[i, idx_cl_op46] > 0.04:
            if txt.count(';') >= 1 or txt.count(':') >= 2:
                P[i, idx_cl_op5] *= 1.12
                P[i, idx_cl_op46] *= 0.90
                claude_boosts += 1
            else:
                P[i, idx_cl_op46] *= 1.10
                P[i, idx_cl_op5] *= 0.92
                claude_boosts += 1

    # 4. OpenAI straight double-quotes signature gate:
    if '"' in txt:
        if P[i, idx_gpt55] > 0.04:
            P[i, idx_gpt55] *= 1.15
            gpt_boosts += 1

        # 5. OpenAI Astra statistical comma-density gate:
    words = len(txt.split())
    if words >= 50:
        comma_rate = txt.count(',') / words * 100
        if comma_rate < 7.5:
            if P[i, idx_astra] > 0.04:
                P[i, idx_astra] *= 1.12
                P[i, idx_terra] *= 0.92
                P[i, idx_luna] *= 0.92
                astra_boosts += 1
        elif comma_rate > 12.0:
            if P[i, idx_astra] > 0.04:
                P[i, idx_astra] *= 0.88
                P[i, idx_terra] *= 1.08
                astra_boosts += 1

    # 6. Kimi vs Qwen Chinese Frontier Gate (Em-dash & Semicolon syntax):
    idx_kimi = LABELS.index('kimi-k3')
    idx_qwen = LABELS.index('qwen3.8-max')
    if '\u2014' in txt or txt.count(';') >= 1:
        if P[i, idx_kimi] > 0.04:
            P[i, idx_kimi] *= 1.14
            P[i, idx_qwen] *= 0.88
            kimi_boosts += 1
    elif '\u2014' not in txt and ';' not in txt:
        if P[i, idx_qwen] > 0.04:
            P[i, idx_qwen] *= 1.10
            P[i, idx_kimi] *= 0.92
            qwen_boosts += 1

    # 7. OpenAI Sibling Fine-Grained Gate (Sol curly quotes vs Luna/Terra):
    idx_sol = LABELS.index('gpt-5.6-sol')
    if ('\u201c' in txt or '\u201d' in txt) and P[i, idx_sol] > 0.04:
        P[i, idx_sol] *= 1.12
        P[i, idx_terra] *= 0.92
        sol_boosts += 1

P /= P.sum(1, keepdims=True)
log(f'sibling hepta-gates: applied {g38_boosts} Gemini, {glm_boosts} GLM, {claude_boosts} Claude, {gpt_boosts} GPT5.5, {astra_boosts} Astra, {kimi_boosts} Kimi, {qwen_boosts} Qwen, {sol_boosts} Sol')

if len(test) >= 18:   # Sinkhorn: equal expected count per model
    for _ in range(100):
        P *= (len(P) / K) / P.sum(0, keepdims=True); P /= P.sum(1, keepdims=True)
score = np.log(P + 1e-12)

GUARDS = [(' \u2014 ', {'claude-opus-4-6', 'claude-opus-5', 'glm-5.3', 'glm-5.3-flash'}),
          ('\u2019', {l for l in LABELS if not l.startswith(('claude', 'glm'))}),
          ('"', {'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'kimi-k3'}),
          ('\u201c', {'gemini-3.8-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'kimi-k3', 'qwen3.8-max'}),
          ('\u2013', {'claude-opus-5', 'gemini-3.1-pro-preview', 'glm-5.3-flash', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'kimi-k3'})]
n_dash = 0
for mark, allowed in GUARDS:   # marks a model never uses in train (zero violations on the 120k pool): act only when the top label is impossible
    forb = np.array([l not in allowed for l in LABELS])
    for i, t in enumerate(test.text.values):
        if mark in t and not allowed.__contains__(LABELS[score[i].argmax()]):
            score[i, forb] = -1e9; n_dash += 1
n_hung = 0
n_anchors = 0
for g in np.unique(groups[(gsize >= 12) & (gsize <= 18)]):   # near-complete prompt groups: each model answers once
    ii = np.where(groups == g)[0]
    g_size = len(ii)
    sub_score = score[ii].copy()
    sub_P = P[ii]
    
    sorted_p = np.sort(sub_P, axis=1)
    margins = sorted_p[:, -1] - sorted_p[:, -2]
    top_p = sorted_p[:, -1]
    top_cls = sub_score.argmax(axis=1)
    
    is_anchor = (margins >= 0.30) & (top_p >= 0.48)
    assigned_in_group = set()
    anchor_local = []
    
    for idx_local in np.argsort(-top_p):
        if is_anchor[idx_local]:
            c = top_cls[idx_local]
            if c not in assigned_in_group:
                assigned_in_group.add(c)
                anchor_local.append(idx_local)
                n_anchors += 1
                
    unassigned_local = [i for i in range(g_size) if i not in anchor_local]
    if unassigned_local:
        avail_cls = [c for c in range(K) if c not in assigned_in_group]
        if len(avail_cls) >= len(unassigned_local):
            cost = -sub_score[unassigned_local][:, avail_cls]
            r_ind, c_ind = linear_sum_assignment(cost)
            for r, c in zip(r_ind, c_ind):
                chosen_cls = avail_cls[c]
                if top_cls[unassigned_local[r]] != chosen_cls:
                    n_hung += 1
                score[ii[unassigned_local[r]]] = -1e9
                score[ii[unassigned_local[r]], chosen_cls] = 0.0
        else:
            r, c = linear_sum_assignment(-sub_score)
            n_hung += int((top_cls[r] != c).sum())
            score[ii[r]] = -1e9; score[ii[r], c] = 0.0
            
    for idx_local in anchor_local:
        c = top_cls[idx_local]
        score[ii[idx_local]] = -1e9
        score[ii[idx_local], c] = 0.0

log(f'typography guards changed {n_dash}, anchor-protected Hungarian: locked {n_anchors} anchors, reassigned {n_hung}')

sub = pd.DataFrame({'id': test.id.values, 'model': [LABELS[i] for i in score.argmax(1)]})
assert sub.model.isin(LABELS).all() and sub.id.is_unique and len(sub) == len(test)
sub.to_parquet(a.out, index=False); log('wrote', a.out)
