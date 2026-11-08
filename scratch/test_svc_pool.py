import time
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

SMALL_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
VIEWS = [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("small", 1, 3, 50000, 0.5)]

LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5', 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']

def shape_text(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)

def small_words_text(t):
    return " ".join(w.lower() if w.lower() in SMALL_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", t))

t0 = time.time()
df = pd.read_parquet('scratch/curated_pool/curated_pool.parquet')
print(f"Loaded {len(df)} rows in {time.time()-t0:.2f}s")

# Filter register_ok == True or take all
if 'register_ok' in df.columns:
    df_clean = df[df['register_ok'] == True].copy()
else:
    df_clean = df.copy()

print(f"Clean rows: {len(df_clean)}")

fit_texts = df_clean.text.astype(str).tolist()[:20000] # Test on 20k rows first
fit_y = df_clean.model.map({l: i for i, l in enumerate(LABELS)}).values[:20000]

print("Fitting TF-IDF and LinearSVC on 20,000 samples...")
t1 = time.time()
A = []
for view, lo, hi, max_features, weight in VIEWS:
    prep = shape_text if view == "shape" else small_words_text if view == "small" else (lambda s: s)
    analyzer = "char" if view == "shape" else "word" if view == "small" else view
    opts = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2,
                max_features=max_features, dtype=np.float32)
    if analyzer == "word":
        opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
    vec = TfidfVectorizer(**opts)
    A.append(vec.fit_transform([prep(t) for t in fit_texts]) * weight)

X = hstack(A, format="csr")
print(f"Feature matrix built: {X.shape} in {time.time()-t1:.2f}s")

t2 = time.time()
svc = LinearSVC(C=0.5, random_state=42, max_iter=2000, class_weight="balanced").fit(X, fit_y)
print(f"LinearSVC fitted in {time.time()-t2:.2f}s")
