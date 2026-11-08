import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, log_loss
from scipy.sparse import hstack
import re

print("=== STARTING LOCAL MODEL ARENA BENCHMARK ===", flush=True)
t0 = time.time()

# 1. Load data
df = pd.read_csv('data/train.csv')
print(f"Loaded {len(df)} samples across {df.model.nunique()} models and {df.prompt_id.nunique()} prompt groups.")

LABELS = sorted(df.model.unique())
K = len(LABELS)
LAB2I = {l: i for i, l in enumerate(LABELS)}
y = df.model.map(LAB2I).values
groups = df.prompt_id.values

# Text preprocessing functions
SMALL_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())

def shape_text(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", str(t))

def small_words_text(t):
    return " ".join(w.lower() if w.lower() in SMALL_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", str(t)))

texts_raw = df.text.astype(str).tolist()
texts_shape = [shape_text(t) for t in texts_raw]
texts_small = [small_words_text(t) for t in texts_raw]

# 2. Setup 5-Fold GroupKFold
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(df, y, groups))

# Define Candidate Feature Set configurations
CANDIDATES = [
    {
        "name": "1. Char N-grams (3-5)",
        "vec": lambda: TfidfVectorizer(ngram_range=(3, 5), analyzer='char', min_df=3, sublinear_tf=True),
        "texts": texts_raw
    },
    {
        "name": "2. Word N-grams (1-2)",
        "vec": lambda: TfidfVectorizer(ngram_range=(1, 2), analyzer='word', min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b|[^\w\s]"),
        "texts": texts_raw
    },
    {
        "name": "3. Word-Shape N-grams (2-5)",
        "vec": lambda: TfidfVectorizer(ngram_range=(2, 5), analyzer='char', min_df=3, sublinear_tf=True),
        "texts": texts_shape
    },
    {
        "name": "4. Function-Words N-grams (1-3)",
        "vec": lambda: TfidfVectorizer(ngram_range=(1, 3), analyzer='word', min_df=2, sublinear_tf=True),
        "texts": texts_small
    },
]

results = []

# Evaluate each candidate standalone
for cand in CANDIDATES:
    oof_preds = np.zeros(len(df))
    t_start = time.time()
    for train_idx, val_idx in folds:
        X_tr = [cand["texts"][i] for i in train_idx]
        X_va = [cand["texts"][i] for i in val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        
        vec = cand["vec"]()
        feat_tr = vec.fit_transform(X_tr)
        feat_va = vec.transform(X_va)
        
        clf = LinearSVC(C=0.5, random_state=42, max_iter=2000)
        clf.fit(feat_tr, y_tr)
        oof_preds[val_idx] = clf.predict(feat_va)
        
    acc = accuracy_score(y, oof_preds)
    elapsed = time.time() - t_start
    print(f"Candidate: {cand['name']:35s} | 5-Fold CV Accuracy: {acc*100:.2f}% | Time: {elapsed:.1f}s")
    results.append({"name": cand["name"], "acc": acc, "oof": oof_preds})

# 3. Test 4-View Full Fusion
print("\n--- Testing 4-View Combined Fusion ---")
t_fusion = time.time()
oof_fusion = np.zeros(len(df))
for train_idx, val_idx in folds:
    y_tr, y_va = y[train_idx], y[val_idx]
    feats_tr, feats_va = [], []
    for cand in CANDIDATES:
        X_tr = [cand["texts"][i] for i in train_idx]
        X_va = [cand["texts"][i] for i in val_idx]
        vec = cand["vec"]()
        feats_tr.append(vec.fit_transform(X_tr))
        feats_va.append(vec.transform(X_va))
    
    A_tr = hstack(feats_tr, format='csr')
    A_va = hstack(feats_va, format='csr')
    
    clf = LinearSVC(C=0.5, random_state=42, max_iter=3000)
    clf.fit(A_tr, y_tr)
    oof_fusion[val_idx] = clf.predict(A_va)

acc_fusion = accuracy_score(y, oof_fusion)
print(f"Candidate: {'5. SOTA 4-View Fusion (Combined)':35s} | 5-Fold CV Accuracy: {acc_fusion*100:.2f}% | Time: {time.time() - t_fusion:.1f}s")

# 4. Error Disagreement Analysis between Char N-grams and Word N-grams
err_char = oof_preds != y
err_fusion = oof_fusion != y
disagree = (oof_preds != oof_fusion).mean()
print(f"\nDisagreement Rate between Word-Only and 4-View: {disagree*100:.2f}%")
print(f"Total time elapsed: {time.time() - t0:.1f}s")
