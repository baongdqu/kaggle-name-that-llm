"""SOTA 120k Stylometric Engine (Champion V30 Apex Frontier Hybrid).
Replaces the 0.563 baseline with V24's 4-View N-gram LinearSVC trained on nlm_pool_v5.parquet."""
import gc, glob, json, os, re, time
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

T0 = time.time()
LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
K = len(LABELS)
LAB2I = {l: i for i, l in enumerate(LABELS)}

DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/test.csv'))
POOL = glob.glob('/kaggle/input/**/nlm_pool_v5.parquet', recursive=True)[0]

test = pd.read_csv(DATA + '/test.csv')
train = pd.read_csv(DATA + '/train.csv')

SMALL_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
VIEWS = [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("small", 1, 3, 50000, 0.5)]

def shape_text(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", str(t))

def small_words_text(t):
    return " ".join(w.lower() if w.lower() in SMALL_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", str(t)))

print(f"Loading pool from {POOL}...", flush=True)
pool = pd.read_parquet(POOL)
fit_texts = train.text.astype(str).tolist() + pool.text.astype(str).tolist()
fit_y = train.model.map(LAB2I).tolist() + pool.model.map(LAB2I).tolist()

te_texts = test.text.astype(str).tolist()

print(f"Fitting 4-View Stylometrics on {len(fit_texts)} samples...", flush=True)
A_fit = []
A_te = []
for view, lo, hi, max_features, weight in VIEWS:
    prep = shape_text if view == "shape" else small_words_text if view == "small" else (lambda s: str(s))
    analyzer = "char" if view == "shape" else "word" if view == "small" else view
    opts = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2,
                max_features=max_features, dtype=np.float32)
    if analyzer == "word":
        opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
    vec = TfidfVectorizer(**opts)
    A_fit.append(vec.fit_transform([prep(t) for t in fit_texts]) * weight)
    A_te.append(vec.transform([prep(t) for t in te_texts]) * weight)
    print(f"  View {view} done: {A_fit[-1].shape[1]} features", flush=True)

X_fit = hstack(A_fit, format="csr")
X_te = hstack(A_te, format="csr")
del A_fit, A_te, fit_texts, pool
gc.collect()

print(f"Fitting LinearSVC on {X_fit.shape}...", flush=True)
svc = LinearSVC(C=0.5, random_state=42, max_iter=2000, class_weight="balanced")
svc.fit(X_fit, fit_y)

# Get decision function (unnormalized logits) on test
LOGIT = svc.decision_function(X_te)
print(f"Stylometrics inference complete! Shape: {LOGIT.shape}", flush=True)

np.save('/kaggle/working/comp_logit.npy', LOGIT)
json.dump({'labels': LABELS, 'ids': test['id'].astype(str).tolist(), 'seconds': time.time() - T0}, open('/kaggle/working/comp_meta.json', 'w'))
print(f"SOTA STYLOMETRIC COMP STACK DONE in {time.time() - T0:.0f}s", flush=True)
