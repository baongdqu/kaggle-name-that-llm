import pandas as pd
import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

t0 = time.time()
print("Loading datasets...")
df_train = pd.read_csv('data/train.csv')
LABELS = sorted(df_train['model'].unique())
ext_df = pd.read_parquet('data/external_short_wide.parquet')

# Build 18-class external dataset
rows = []
for m in LABELS:
    if m in ext_df.columns:
        s = ext_df[['id', m]].dropna().rename(columns={m: 'text'})
        s['model'] = m
        rows.append(s)
cat = pd.concat(rows, ignore_index=True)

all_texts = df_train['text'].tolist() + cat['text'].astype(str).tolist()
all_labels = [m for m in df_train['model']] + [m for m in cat['model']]
y_all = np.array([{l: i for i, l in enumerate(LABELS)}[m] for m in all_labels])

print(f"Total dataset: {len(all_texts)} rows across {len(LABELS)} classes. Prep time: {time.time()-t0:.1f}s")

# Vectorize with shape and function n-grams
def shape_text(s):
    res = []
    for c in s:
        if c.isupper():
            res.append('X')
        elif c.islower():
            res.append('x')
        elif c.isdigit():
            res.append('0')
        elif c.isspace():
            res.append(' ')
        else:
            res.append(c)
    return ''.join(res)

import re
STOP = set("""a about above after again against all am an and any are aren't as at be because been before being below
between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from
further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how
how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of
off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't
so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're
they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's
when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're
you've""".split())

def function_text(s):
    toks = re.findall(r"(?u)\b\w+\b|[^\w\s]", s)
    return ' '.join([t.lower() if t.lower() in STOP or not t.isalnum() else 'CONTENT' for t in toks])

t_feat = time.time()
A = []
# Use fast char and shape n-grams
v_char = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True)
A.append(v_char.fit_transform(all_texts))

v_shape = TfidfVectorizer(analyzer='char', ngram_range=(2, 6), min_df=2, max_features=20000, sublinear_tf=True)
A.append(v_shape.fit_transform([shape_text(t) for t in all_texts]))

X = hstack(A, format='csr')
print(f"Features created: shape {X.shape} in {time.time()-t_feat:.1f}s")

t_train = time.time()
svc = LinearSVC(C=0.3, random_state=42, max_iter=2000).fit(X, y_all)
print(f"LinearSVC fit completed in {time.time()-t_train:.1f}s! Total classes: {len(svc.classes_)}")
