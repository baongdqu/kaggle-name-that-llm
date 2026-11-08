import numpy as np, pandas as pd
from collections import Counter

train = pd.read_csv('data/train.csv')

pairs = [
    ('gemini-3.7-flash', 'gemini-3.8-flash'),
    ('gpt-5.6-sol', 'gpt-5.6-terra'),
    ('glm-5.3', 'glm-5.3-flash'),
    ('claude-fable-5', 'glm-5.3-flash'),
    ('claude-opus-4-8', 'claude-opus-4-6'),
    ('claude-opus-5', 'claude-opus-4-8')
]

for m1, m2 in pairs:
    df1 = train[train['model'] == m1].set_index('prompt_id')
    df2 = train[train['model'] == m2].set_index('prompt_id')
    common = df1.index.intersection(df2.index)
    
    w1 = [len(df1.loc[p, 'text'].split()) for p in common]
    w2 = [len(df2.loc[p, 'text'].split()) for p in common]
    
    # Punctuation counts
    dashes1 = [df1.loc[p, 'text'].count('—') for p in common]
    dashes2 = [df2.loc[p, 'text'].count('—') for p in common]
    
    colons1 = [df1.loc[p, 'text'].count(':') for p in common]
    colons2 = [df2.loc[p, 'text'].count(':') for p in common]
    
    semis1 = [df1.loc[p, 'text'].count(';') for p in common]
    semis2 = [df2.loc[p, 'text'].count(';') for p in common]

    curly1 = [df1.loc[p, 'text'].count('’') + df1.loc[p, 'text'].count('“') for p in common]
    curly2 = [df2.loc[p, 'text'].count('’') + df2.loc[p, 'text'].count('“') for p in common]

    print(f"\n==================== {m1} vs {m2} ====================")
    print(f"Words: {np.mean(w1):.1f} vs {np.mean(w2):.1f} ({m1} > {m2}: {sum(a>b for a,b in zip(w1,w2))}/150)")
    print(f"Em-Dash (—): {np.mean(dashes1):.2f} vs {np.mean(dashes2):.2f}")
    print(f"Colon (:): {np.mean(colons1):.2f} vs {np.mean(colons2):.2f}")
    print(f"Semicolon (;): {np.mean(semis1):.2f} vs {np.mean(semis2):.2f}")
    print(f"Curly Quotes: {np.mean(curly1):.2f} vs {np.mean(curly2):.2f}")

