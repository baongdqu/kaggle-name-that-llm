import time
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
import re

train = pd.read_csv('data/train.csv')
print(f"Loaded train.csv: {len(train)} rows")

LABELS = sorted(train.model.unique())
y = train.model.map({l: i for i, l in enumerate(LABELS)}).values

SMALL_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
VIEWS = [("char", 3, 5, 50000, 1.0), ("word", 1, 3, 30000, 0.7), ("shape", 2, 6, 20000, 0.5), ("small", 1, 3, 20000, 0.5)]

def shape_text(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)

def small_words_text(t):
    return " ".join(w.lower() if w.lower() in SMALL_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", t))

sgkf = StratifiedGroupKFold(n_splits=5)

def evaluate_pool(pool_name, pool_df):
    t0 = time.time()
    oof_preds = np.zeros(len(train))
    
    # Pre-extract pool features
    pool_texts = pool_df.text.tolist()
    pool_y = pool_df.model.map({l: i for i, l in enumerate(LABELS)}).values
    
    train_texts = train.text.tolist()
    
    A_pool, A_train = [], []
    for view, lo, hi, max_features, weight in VIEWS:
        prep = shape_text if view == "shape" else small_words_text if view == "small" else (lambda s: s)
        analyzer = "char" if view == "shape" else "word" if view == "small" else view
        opts = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2,
                    max_features=max_features, dtype=np.float32)
        if analyzer == "word":
            opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        vec = TfidfVectorizer(**opts)
        # fit on pool + train
        all_texts = [prep(t) for t in (pool_texts + train_texts)]
        vec.fit(all_texts)
        A_pool.append(vec.transform([prep(t) for t in pool_texts]) * weight)
        A_train.append(vec.transform([prep(t) for t in train_texts]) * weight)
        
    X_pool = hstack(A_pool, format="csr")
    X_train = hstack(A_train, format="csr")
    
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(train, train.model, train.prompt_id)):
        # Fit on pool + trn_idx
        X_fit = hstack([X_pool.T, X_train[trn_idx].T], format="csr").T
        y_fit = np.r_[pool_y, y[trn_idx]]
        
        svc = LinearSVC(C=0.5, random_state=42, max_iter=2000, class_weight="balanced")
        svc.fit(X_fit, y_fit)
        
        preds = svc.predict(X_train[val_idx])
        oof_preds[val_idx] = preds
        
    ba = balanced_accuracy_score(y, oof_preds)
    print(f"Pool: {pool_name} ({len(pool_df)} rows) -> 5-Fold Grouped CV BA: {ba:.4f} in {time.time()-t0:.1f}s")
    return ba

# Evaluate pool_v4
pool_v4 = pd.read_parquet('c:/Users/s3cr3t/My Drive (baongdqu@gmail.com)/zzz kỹ năng tech - ai (separator)/( project ) kaggle competition name that label/scratch/prvsiyan/pool_v4.parquet')
evaluate_pool("pool_v4 (75k)", pool_v4)

# Evaluate curated_pool
curated = pd.read_parquet('c:/Users/s3cr3t/My Drive (baongdqu@gmail.com)/zzz kỹ năng tech - ai (separator)/( project ) kaggle competition name that label/scratch/curated_pool/curated_pool.parquet')
curated_clean = curated[curated["register_ok"] == True].copy()
evaluate_pool("curated_pool (81k clean)", curated_clean)
