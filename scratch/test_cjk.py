import pandas as pd
import numpy as np

df = pd.read_csv('data/train.csv')
print('=== Unique Models ===')
print(sorted(df['model'].unique()))

# Check Chinese characters count across all models
import re
def count_cjk(s):
    return len(re.findall(r'[\u4e00-\u9fff]', s))

print('\n=== CJK Character Count by Model ===')
for m in sorted(df['model'].unique()):
    sub = df[df['model'] == m]['text']
    cjk = sub.apply(count_cjk)
    if cjk.sum() > 0:
        print(f'{m:25s} | total CJK: {cjk.sum()} | max CJK: {cjk.max()} | rows with CJK: {(cjk > 0).sum()}/{len(sub)}')
