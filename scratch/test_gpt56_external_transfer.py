import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

df_ext = pd.read_parquet('data/external_short_wide.parquet')
train = pd.read_csv('data/train.csv')

gpt_models = ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra']

# Extract clean external samples
rows = []
for m in gpt_models:
    sub = df_ext[df_ext[m].notna()][['prompt', m]].copy()
    sub['model'] = m
    sub.rename(columns={m: 'text'}, inplace=True)
    rows.append(sub)

ext_gpt = pd.concat(rows, ignore_index=True)
# filter short
ext_gpt = ext_gpt[ext_gpt['text'].str.split().str.len() >= 30].reset_index(drop=True)
print("External GPT-5.6 counts:\n", ext_gpt['model'].value_counts())

# Official train subset
sub_val = train[train['model'].isin(gpt_models)].copy().reset_index(drop=True)
print("\nOfficial Val GPT-5.6 counts:\n", sub_val['model'].value_counts())

y_train = np.array([gpt_models.index(m) for m in ext_gpt['model'].values])
y_val = np.array([gpt_models.index(m) for m in sub_val['model'].values])

v1 = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=35000, sublinear_tf=True)
v2 = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', max_features=25000, sublinear_tf=True)

X_tr = hstack([v1.fit_transform(ext_gpt['text'].values), v2.fit_transform(ext_gpt['text'].values)], format='csr')
X_va = hstack([v1.transform(sub_val['text'].values), v2.transform(sub_val['text'].values)], format='csr')

svc = LinearSVC(C=0.3, random_state=42, max_iter=2000, dual=True)
svc.fit(X_tr, y_train)
preds = svc.predict(X_va)

acc = accuracy_score(y_val, preds)
print(f"\n🔥 Zero-Shot Transfer Accuracy on 3 GPT-5.6 Siblings: {acc*100:.2f}% (vs 33.33% random guessing)!")

