import pandas as pd
import numpy as np

df = pd.read_csv('data/train.csv')
models = sorted(df['model'].unique())

print("=== ZERO-FREQUENCY DISCOVERIES ACROSS 18 MODELS ===")

checks = [
    ("Contains curly apostrophe ’", lambda t: "’" in t),
    ("Contains straight apostrophe '", lambda t: "'" in t),
    ("Contains em-dash —", lambda t: "—" in t),
    ("Contains en-dash –", lambda t: "–" in t),
    ("Contains curly open quote “", lambda t: "“" in t),
    ("Contains straight double quote \"", lambda t: '"' in t),
    ("Contains semicolon ;", lambda t: ";" in t),
    ("Contains colon :", lambda t: ":" in t),
    ("Contains hyphen -", lambda t: "-" in t),
    ("Contains parentheses ()", lambda t: "(" in t and ")" in t),
    ("Contains square brackets []", lambda t: "[" in t and "]" in t),
    ("Contains asterisks *", lambda t: "*" in t),
    ("Contains ellipsis …", lambda t: "…" in t),
    ("Contains triple dot ...", lambda t: "..." in t),
    ("Contains markdown bold **", lambda t: "**" in t),
    ("Contains markdown heading #", lambda t: "#" in t),
]

for label, fn in checks:
    counts = df.groupby('model')['text'].apply(lambda s: s.apply(fn).sum())
    zeros = counts[counts == 0].index.tolist()
    if zeros:
        print(f"\n[FEATURE]: {label}")
        print(f"  -> NEVER APPEARS (0 out of 150) in: {zeros}")
