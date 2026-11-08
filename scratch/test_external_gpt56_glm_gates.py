import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score, classification_report

print("Loading data...", flush=True)
train = pd.read_csv('data/train.csv')
ext = pd.read_parquet('data/external_short_wide.parquet')

# --- 1. Evaluate GPT-5.6 Family (Luna, Sol, Terra) ---
gpt_models = ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra']

# Extract clean external samples
ext_rows = []
for m in gpt_models:
    sub = ext[ext[m].notna()][['id', m]].copy()
    sub.rename(columns={m: 'text'}, inplace=True)
    sub['model'] = m
    # Filter out empty or very short passages
    sub = sub[sub['text'].str.len() > 50]
    ext_rows.append(sub)

df_gpt_ext = pd.concat(ext_rows, ignore_index=True)
print(f"GPT-5.6 External Samples: {len(df_gpt_ext)} ({df_gpt_ext['model'].value_counts().to_dict()})")

y_gpt_train = np.array([gpt_models.index(m) for m in df_gpt_ext['model'].values])

# Train TF-IDF vectorizers
v1 = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=35000, sublinear_tf=True, dtype=np.float32)
v2 = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', max_features=25000, sublinear_tf=True, dtype=np.float32)

X_gpt_tr = hstack([v1.fit_transform(df_gpt_ext['text'].values), v2.fit_transform(df_gpt_ext['text'].values)], format='csr')

# Evaluate on official train.csv for GPT-5.6 models
sub_train_gpt = train[train['model'].isin(gpt_models)].copy().reset_index(drop=True)
y_gpt_val = np.array([gpt_models.index(m) for m in sub_train_gpt['model'].values])
X_gpt_va = hstack([v1.transform(sub_train_gpt['text'].values), v2.transform(sub_train_gpt['text'].values)], format='csr')

clf_gpt = LinearSVC(C=0.5, random_state=42, max_iter=2000, dual=True)
clf_gpt.fit(X_gpt_tr, y_gpt_train)
preds_gpt = clf_gpt.predict(X_gpt_va)

bacc_gpt = balanced_accuracy_score(y_gpt_val, preds_gpt)
print(f"\n🎯 [GPT-5.6 External Gate] Balanced Accuracy on train.csv: {bacc_gpt*100:.2f}%")
print(classification_report(y_gpt_val, preds_gpt, target_names=gpt_models))

# --- 2. Evaluate GLM Family (GLM-5.3, GLM-5.3-flash) ---
glm_models = ['glm-5.3', 'glm-5.3-flash']
ext_glm_rows = []
for m in glm_models:
    sub = ext[ext[m].notna()][['id', m]].copy()
    sub.rename(columns={m: 'text'}, inplace=True)
    sub['model'] = m
    sub = sub[sub['text'].str.len() > 50]
    ext_glm_rows.append(sub)

df_glm_ext = pd.concat(ext_glm_rows, ignore_index=True)
print(f"GLM External Samples: {len(df_glm_ext)} ({df_glm_ext['model'].value_counts().to_dict()})")

y_glm_train = np.array([glm_models.index(m) for m in df_glm_ext['model'].values])

v_glm1 = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', max_features=35000, sublinear_tf=True, dtype=np.float32)
v_glm2 = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', max_features=25000, sublinear_tf=True, dtype=np.float32)

X_glm_tr = hstack([v_glm1.fit_transform(df_glm_ext['text'].values), v_glm2.fit_transform(df_glm_ext['text'].values)], format='csr')

sub_train_glm = train[train['model'].isin(glm_models)].copy().reset_index(drop=True)
y_glm_val = np.array([glm_models.index(m) for m in sub_train_glm['model'].values])
X_glm_va = hstack([v_glm1.transform(sub_train_glm['text'].values), v_glm2.transform(sub_train_glm['text'].values)], format='csr')

clf_glm = LinearSVC(C=0.5, random_state=42, max_iter=2000, dual=True)
clf_glm.fit(X_glm_tr, y_glm_train)
preds_glm = clf_glm.predict(X_glm_va)

bacc_glm = balanced_accuracy_score(y_glm_val, preds_glm)
print(f"\n🎯 [GLM External Gate] Balanced Accuracy on train.csv: {bacc_glm*100:.2f}%")
print(classification_report(y_glm_val, preds_glm, target_names=glm_models))

