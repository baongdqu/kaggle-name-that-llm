import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv')
g37 = train[train['model'] == 'gemini-3.7-flash'].set_index('prompt_id')
g38 = train[train['model'] == 'gemini-3.8-flash'].set_index('prompt_id')

common_pids = g37.index.intersection(g38.index)
print(f"Total common prompts for gemini-3.7 vs 3.8: {len(common_pids)}")

diffs = []
for pid in common_pids[:10]:
    t37 = g37.loc[pid, 'text']
    t38 = g38.loc[pid, 'text']
    print(f"\n--- PROMPT {pid} ---")
    print(f"G3.7 ({len(t37.split())} words): {t37[:160]}...")
    print(f"G3.8 ({len(t38.split())} words): {t38[:160]}...")
    diffs.append(len(t38.split()) - len(t37.split()))

print(f"\nWord length diff (G3.8 - G3.7) over first 10: mean = {np.mean(diffs):.2f}")

