import pandas as pd
import numpy as np

# Load train to see how distinguishable models are
df = pd.read_csv('data/train.csv')
print('Total rows:', len(df))

# Let's test comma density for gpt-6-astra on texts >= 50 words
sub_oa = df[df['model'].isin(['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra'])].copy()
sub_oa['words'] = sub_oa['text'].apply(lambda t: len(t.split()))
sub_oa = sub_oa[sub_oa['words'] >= 50]
sub_oa['comma_rate'] = sub_oa.apply(lambda r: r['text'].count(',') / r['words'] * 100, axis=1)

print('=== OpenAI Comma Rate Analysis (words >= 50) ===')
for m in ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra']:
    c = sub_oa[sub_oa['model'] == m]['comma_rate']
    low_rate = (c < 7.5).mean()
    high_rate = (c > 11.5).mean()
    print(f'{m:15s} | mean: {c.mean():.2f} | < 7.5: {low_rate*100:.1f}% | > 11.5: {high_rate*100:.1f}%')
