import pandas as pd
import numpy as np

df = pd.read_csv('data/train.csv')
print(f"Total training samples: {len(df)}")

models = sorted(df['model'].unique())

# Define typographical markers
markers = {
    "curly_apos_’": lambda t: "’" in t,
    "straight_apos_'": lambda t: "'" in t,
    "curly_quote_“": lambda t: "“" in t or "”" in t,
    "straight_quote_\"": lambda t: '"' in t,
    "em_dash_—": lambda t: "—" in t,
    "en_dash_–": lambda t: "–" in t,
    "ellipsis_…": lambda t: "…" in t,
    "triple_dot_...": lambda t: "..." in t,
    "colon_ratio": lambda t: t.count(":") / max(len(t.split()), 1),
    "semicolon_;": lambda t: ";" in t,
    "chinese_comma_，": lambda t: "，" in t,
    "chinese_period_。": lambda t: "。" in t,
}

stats = {}
for m in models:
    sub = df[df['model'] == m]
    row = {}
    for name, fn in markers.items():
        if "ratio" in name:
            row[name] = round(sub['text'].apply(fn).mean(), 4)
        else:
            pct = round(sub['text'].apply(fn).mean() * 100, 1)
            row[name] = f"{pct}%"
    stats[m] = row

res_df = pd.DataFrame(stats).T
print(res_df.to_string())
