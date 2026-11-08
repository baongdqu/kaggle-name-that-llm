import pandas as pd, numpy as np

train = pd.read_csv('data/train.csv')
ext = pd.read_parquet('data/external_short_wide.parquet')

print("--- OFFICIAL TRAIN.CSV ---")
print("Sol words:", [len(t.split()) for t in train[train['model']=='gpt-5.6-sol']['text'][:5]])
print("Terra words:", [len(t.split()) for t in train[train['model']=='gpt-5.6-terra']['text'][:5]])

print("\n--- EXTERNAL DATASET ---")
sol_e = ext['gpt-5.6-sol'].dropna().tolist()
terra_e = ext['gpt-5.6-terra'].dropna().tolist()
print("Sol external mean words:", np.mean([len(t.split()) for t in sol_e[:100]]))
print("Terra external mean words:", np.mean([len(t.split()) for t in terra_e[:100]]))

print("Sol external sample:\n", sol_e[0][:200])
print("Terra external sample:\n", terra_e[0][:200])

