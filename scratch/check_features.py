import pandas as pd
df = pd.read_csv('data/train.csv')

print("\n--- Claude Family Detailed ---")
for m in ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5']:
    sub = df[df['model'] == m]
    words = sub['text'].apply(lambda x: len(x.split())).mean()
    commas = sub['text'].apply(lambda x: x.count(',') / len(x.split()) * 100).mean()
    semis = sub['text'].apply(lambda x: x.count(';')).mean()
    colons = sub['text'].apply(lambda x: x.count(':')).mean()
    em_dash = sub['text'].apply(lambda x: x.count('\u2014')).mean()
    bullet = sub['text'].apply(lambda x: any(b in x for b in ['\n- ', '\n* ', '\n1. '])).mean()
    paragraphs = sub['text'].apply(lambda x: len(x.split('\n\n'))).mean()
    print(f"{m:16s} | Words: {words:.1f} | Paras: {paragraphs:.2f} | Semis: {semis:.2f} | Colons: {colons:.2f} | Em: {em_dash:.2f} | Bullet%: {bullet:.2f}")
