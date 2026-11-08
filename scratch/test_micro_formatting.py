import pandas as pd
import numpy as np
import re
import time

t0 = time.time()
print("Loading train.csv...")
df = pd.read_csv('data/train.csv')

def extract_micro_features(texts):
    feats = []
    for t in texts:
        L = len(t) + 1
        paras = [p for p in t.split('\n\n') if p.strip()]
        p_lens = [len(p) for p in paras] or [0]
        
        f = [
            sum(c.isspace() for c in t) / L,
            t.count('\n') / L,
            t.count('\n\n'),
            len(paras),
            float(np.mean(p_lens)),
            float(np.std(p_lens)),
            t.count('—'),
            t.count('–'),
            t.count('-'),
            t.count('…'),
            t.count('...'),
            len(re.findall(r'\*\*.*?\*\*', t)),
            len(re.findall(r'(?<!\*)\*(?!\*).*?(?<!\*)\*(?!\*)', t)),
            len(re.findall(r'(?m)^-\s+', t)),
            len(re.findall(r'(?m)^\*\s+', t)),
            len(re.findall(r'(?m)^\d+\.\s+', t)),
            t.count('```'),
            len(re.findall(r'`[^`]+`', t)),
            len(re.findall(r'(?m)^>\s+', t)),
            t.count('“') + t.count('”'),
            t.count('"'),
            t.count(':'),
            t.count(';'),
            t.count('!'),
            t.count('?'),
            len(re.findall(r'(?i)\bhowever\b', t)),
            len(re.findall(r'(?i)\bfurthermore\b', t)),
            len(re.findall(r'(?i)\bmoreover\b', t)),
            len(re.findall(r'(?i)\btherefore\b', t)),
            len(re.findall(r'(?m)^this\b', t, re.IGNORECASE))
        ]
        feats.append(f)
    return np.array(feats, dtype=np.float32)

X_micro = extract_micro_features(df['text'].tolist())
print(f"Extracted {X_micro.shape[1]} micro-features for {X_micro.shape[0]} rows in {time.time()-t0:.2f}s!")

# Check em-dash per model
df['em_dash'] = X_micro[:, 6]
df['bold_md'] = X_micro[:, 11]
df['double_nl'] = X_micro[:, 2]

print("\nTop 5 models using em-dash '—':")
print(df.groupby('model')['em_dash'].mean().sort_values(ascending=False).head(5))

print("\nTop 5 models using bold '**':")
print(df.groupby('model')['bold_md'].mean().sort_values(ascending=False).head(5))

print("\nTop 5 models using double newline '\\n\\n':")
print(df.groupby('model')['double_nl'].mean().sort_values(ascending=False).head(5))
