import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

train = pd.read_csv('data/train.csv')
pids = train['prompt_id'].unique()

sub37 = train[train['model'] == 'gemini-3.7-flash'].set_index('prompt_id')
sub38 = train[train['model'] == 'gemini-3.8-flash'].set_index('prompt_id')

common = sub37.index.intersection(sub38.index)

# Compare word counts directly
w37 = [len(sub37.loc[p, 'text'].split()) for p in common]
w38 = [len(sub38.loc[p, 'text'].split()) for p in common]

c37 = [len(sub37.loc[p, 'text']) for p in common]
c38 = [len(sub38.loc[p, 'text']) for p in common]

print(f"Across 150 prompts:")
print(f"Mean words G3.7: {np.mean(w37):.1f} | Mean words G3.8: {np.mean(w38):.1f}")
print(f"Mean chars G3.7: {np.mean(c37):.1f} | Mean chars G3.8: {np.mean(c38):.1f}")

# How often is G3.7 longer than G3.8 within the exact same prompt?
g37_longer = sum(w1 > w2 for w1, w2 in zip(w37, w38))
g38_longer = sum(w2 > w1 for w1, w2 in zip(w37, w38))
ties = sum(w1 == w2 for w1, w2 in zip(w37, w38))
print(f"Prompt-by-prompt head-to-head word length:")
print(f"G3.7 > G3.8: {g37_longer}/150 ({g37_longer/150*100:.1f}%)")
print(f"G3.8 > G3.7: {g38_longer}/150 ({g38_longer/150*100:.1f}%)")
print(f"Ties: {ties}/150")

# Character counts head-to-head
c37_longer = sum(c1 > c2 for c1, c2 in zip(c37, c38))
c38_longer = sum(c2 > c1 for c1, c2 in zip(c37, c38))
print(f"\nCharacter length head-to-head:")
print(f"G3.7 > G3.8: {c37_longer}/150 ({c37_longer/150*100:.1f}%)")
print(f"G3.8 > G3.7: {c38_longer}/150 ({c38_longer/150*100:.1f}%)")

