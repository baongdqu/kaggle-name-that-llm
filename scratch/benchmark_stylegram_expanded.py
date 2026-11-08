import time
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.svm import LinearSVC

t0 = time.time()
print("1. Loading train and external data...")
train = pd.read_csv('data/train.csv')
LABELS = sorted(train['model'].unique())
K = len(LABELS)

ext_df = pd.read_parquet('data/external_short_wide.parquet')

# Build balanced external rows: max 500 per class to keep speed fast & distribution balanced
ext_rows = []
for m in LABELS:
    if m in ext_df.columns:
        sub = ext_df[['id', m]].dropna().rename(columns={m: 'text'})
        sub['model'] = m
        if len(sub) > 600:
            sub = sub.sample(600, random_state=42)
        ext_rows.append(sub)

ext_data = pd.concat(ext_rows, ignore_index=True)
print(f"External selected rows: {len(ext_data)} across {ext_data['model'].nunique()} classes")

combined_train = pd.concat([train[['text', 'model']], ext_data[['text', 'model']]], ignore_index=True)
print(f"Total training data: {len(combined_train)} rows")
print(combined_train['model'].value_counts())

TR = combined_train['text'].astype(str).tolist()
y = combined_train['model'].map({l: i for i, l in enumerate(LABELS)}).values

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())

def shape_text(text):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", text)

def function_text(text):
    return " ".join(t.lower() if t.lower() in FUNCTION_WORDS or not t.isalnum() else "CONTENT"
                    for t in re.findall(r"\w+|[^\w\s]", text))

SPEC = [('char', 3, 5, False, 100000, 1.0), ('word', 1, 3, False, 80000, 0.7),
        ('shape', 2, 6, False, 40000, 0.5), ('function', 1, 3, False, 30000, 0.5)]

print("2. Vectorizing...")
t_vec = time.time()
A = []
for analyzer, lo, hi, lower, maxf, w in SPEC:
    tf = shape_text if analyzer == 'shape' else function_text if analyzer == 'function' else str
    actual = 'char' if analyzer == 'shape' else 'word' if analyzer == 'function' else analyzer
    opts = dict(analyzer=actual, ngram_range=(lo, hi), lowercase=lower, sublinear_tf=True, min_df=2,
                max_features=maxf, dtype=np.float32)
    if actual == 'word':
        opts['token_pattern'] = r"(?u)\b\w+\b|[^\w\s]"
    v = TfidfVectorizer(**opts)
    A.append(v.fit_transform([tf(t) for t in TR]) * w)
A_STYLE = hstack(A, format='csr')
print(f"Vectorization done in {time.time() - t_vec:.1f}s, shape: {A_STYLE.shape}")

print("3. Fitting LinearSVC...")
t_fit = time.time()
svc = LinearSVC(C=0.3, random_state=42, max_iter=2000).fit(A_STYLE, y)
print(f"LinearSVC fit completed in {time.time() - t_fit:.1f}s!")
print(f"Total benchmark time: {time.time() - t0:.1f}s")
