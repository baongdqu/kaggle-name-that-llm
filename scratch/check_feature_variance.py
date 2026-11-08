import pandas as pd
import numpy as np
import re

df = pd.read_csv('data/train.csv')

def check_feature(name, fn):
    vals = [fn(t) for t in df['text']]
    df[name] = vals
    means = df.groupby('model')[name].mean()
    print(f"=== {name} (min={means.min():.4f}, max={means.max():.4f}, std={means.std():.4f}) ===")
    print(means.sort_values(ascending=False).head(4).to_dict())
    print("Bottom 3:", means.sort_values().head(3).to_dict())

check_feature('em_dash', lambda t: t.count('—'))
check_feature('semicolon', lambda t: t.count(';'))
check_feature('colon', lambda t: t.count(':'))
check_feature('comma_rate', lambda t: t.count(',') / (len(t) + 1))
check_feature('hyphen_rate', lambda t: t.count('-') / (len(t) + 1))
check_feature('avg_word_len', lambda t: np.mean([len(w) for w in t.split()] or [0]))
check_feature('however_count', lambda t: len(re.findall(r'(?i)\bhowever\b', t)))
check_feature('furthermore_count', lambda t: len(re.findall(r'(?i)\bfurthermore\b', t)))
check_feature('curly_quotes', lambda t: t.count('“') + t.count('”'))
check_feature('straight_quotes', lambda t: t.count('"'))
