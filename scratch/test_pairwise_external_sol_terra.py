import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train = pd.read_csv('data/train.csv')
df_ext = pd.read_parquet('data/external_short_wide.parquet')

# Sol vs Terra in train (validation target)
sub_train = train[train['model'].isin(['gpt-5.6-sol', 'gpt-5.6-terra'])].copy().reset_index(drop=True)
y_val = (sub_train['model'] == 'gpt-5.6-sol').astype(int).values
text_val = sub_train['text'].values
print(f"Validation official samples (Sol vs Terra): {len(sub_train)}")

# Sol vs Terra in external (training target)
sol_ext = df_ext['gpt-5.6-sol'].dropna().tolist()
terra_ext = df_ext['gpt-5.6-terra'].dropna().tolist()

text_train = sol_ext + terra_ext
y_train = np.array([1]*len(sol_ext) + [0]*len(terra_ext))
print(f"External training samples: {len(sol_ext)} Sol + {len(terra_ext)} Terra = {len(text_train)} total")

v = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=40000, sublinear_tf=True)
X_train = v.fit_transform(text_train)
X_val = v.transform(text_val)

clf = LogisticRegression(C=1.0, max_iter=1000)
clf.fit(X_train, y_train)
preds = clf.predict(X_val)

acc = accuracy_score(y_val, preds)
print(f"\n🔥 Dedicated Sol vs Terra Binary Accuracy on official data: {acc*100:.2f}% (vs 50% random)!")

