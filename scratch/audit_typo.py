import numpy as np
import pandas as pd

train = pd.read_csv('data/train.csv')
print(f"Loaded train: {len(train)} rows")

curly_apos = train.text.str.contains('’').sum()
em_dash = train.text.str.contains('—').sum()
straight_quote = train.text.str.contains('"').sum()

print(f"Passages with '’' (curly apostrophe): {curly_apos} ({curly_apos/len(train)*100:.1f}%)")
print(f"Passages with '—' (em-dash): {em_dash} ({em_dash/len(train)*100:.1f}%)")
print(f"Passages with '\"' (straight quote): {straight_quote} ({straight_quote/len(train)*100:.1f}%)")

print("\nDistribution of '’' across models:")
print(train[train.text.str.contains('’')]['model'].value_counts())

print("\nDistribution of '—' across models:")
print(train[train.text.str.contains('—')]['model'].value_counts())

print("\nDistribution of '\"' across models:")
print(train[train.text.str.contains('"')]['model'].value_counts())
