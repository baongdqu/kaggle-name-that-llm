import numpy as np, pandas as pd, re

train = pd.read_csv('data/train.csv')
pids = train['prompt_id'].unique()

print("="*70)
print("🔍 1. Forensic Signatures: claude-fable-5 vs glm-5.3 vs glm-5.3-flash")
print("="*70)

sub_fable = train[train['model'].isin(['claude-fable-5', 'glm-5.3', 'glm-5.3-flash'])]

features = {}
for m in ['claude-fable-5', 'glm-5.3', 'glm-5.3-flash']:
    df_m = sub_fable[sub_fable['model'] == m].copy()
    texts = df_m['text'].astype(str).tolist()
    wlens = [len(t.split()) for t in texts]
    
    # Check key punctuations and tokens
    punc_counts = {
        'avg_words': np.mean(wlens),
        'colon (:)': np.mean([t.count(':') / (len(t.split())+1) * 1000 for t in texts]),
        'semicolon (;)': np.mean([t.count(';') / (len(t.split())+1) * 1000 for t in texts]),
        'em_dash (—)': np.mean([t.count('—') / (len(t.split())+1) * 1000 for t in texts]),
        'en_dash (–)': np.mean([t.count('–') / (len(t.split())+1) * 1000 for t in texts]),
        'hyphen (-)': np.mean([t.count('-') / (len(t.split())+1) * 1000 for t in texts]),
        'curly_dquote (“)': np.mean([t.count('“') / (len(t.split())+1) * 1000 for t in texts]),
        'curly_squote (’ )': np.mean([t.count('’') / (len(t.split())+1) * 1000 for t in texts]),
        'straight_dquote (")': np.mean([t.count('"') / (len(t.split())+1) * 1000 for t in texts]),
        'straight_squote (\')': np.mean([t.count("'") / (len(t.split())+1) * 1000 for t in texts]),
        'parens (()': np.mean([t.count('(') / (len(t.split())+1) * 1000 for t in texts]),
        'slash (/)': np.mean([t.count('/') / (len(t.split())+1) * 1000 for t in texts]),
        'exclamation (!)': np.mean([t.count('!') / (len(t.split())+1) * 1000 for t in texts]),
        'question (?)': np.mean([t.count('?') / (len(t.split())+1) * 1000 for t in texts]),
        'asterisk (*)': np.mean([t.count('*') / (len(t.split())+1) * 1000 for t in texts]),
        'backtick (`)': np.mean([t.count('`') / (len(t.split())+1) * 1000 for t in texts]),
        'triple_asterisk (***)': np.mean([t.count('***') for t in texts]),
        'markdown_header (#)': np.mean([t.count('#') / (len(t.split())+1) * 1000 for t in texts]),
    }
    features[m] = punc_counts

df_feat = pd.DataFrame(features)
print(df_feat.to_string())

print("\n" + "="*70)
print("🔍 2. Forensic Signatures: gemini-3.6-flash vs gemini-3.7-flash vs gemini-3.8-flash")
print("="*70)

sub_gem = train[train['model'].isin(['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'])]
gem_feat = {}
for m in ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash']:
    df_m = sub_gem[sub_gem['model'] == m].copy()
    texts = df_m['text'].astype(str).tolist()
    wlens = [len(t.split()) for t in texts]
    gem_feat[m] = {
        'avg_words': np.mean(wlens),
        'colon (:)': np.mean([t.count(':') / (len(t.split())+1) * 1000 for t in texts]),
        'em_dash (—)': np.mean([t.count('—') / (len(t.split())+1) * 1000 for t in texts]),
        'curly_dquote (“)': np.mean([t.count('“') / (len(t.split())+1) * 1000 for t in texts]),
        'curly_squote (’ )': np.mean([t.count('’') / (len(t.split())+1) * 1000 for t in texts]),
        'straight_dquote (")': np.mean([t.count('"') / (len(t.split())+1) * 1000 for t in texts]),
        'parens (()': np.mean([t.count('(') / (len(t.split())+1) * 1000 for t in texts]),
        'asterisk (*)': np.mean([t.count('*') / (len(t.split())+1) * 1000 for t in texts]),
        'exclamation (!)': np.mean([t.count('!') / (len(t.split())+1) * 1000 for t in texts]),
    }
print(pd.DataFrame(gem_feat).to_string())

print("\n" + "="*70)
print("🔍 3. Forensic Signatures: gpt-5.6-luna vs gpt-5.6-sol vs gpt-5.6-terra")
print("="*70)

sub_gpt = train[train['model'].isin(['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra'])]
gpt_feat = {}
for m in ['gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra']:
    df_m = sub_gpt[sub_gpt['model'] == m].copy()
    texts = df_m['text'].astype(str).tolist()
    wlens = [len(t.split()) for t in texts]
    gpt_feat[m] = {
        'avg_words': np.mean(wlens),
        'colon (:)': np.mean([t.count(':') / (len(t.split())+1) * 1000 for t in texts]),
        'em_dash (—)': np.mean([t.count('—') / (len(t.split())+1) * 1000 for t in texts]),
        'curly_dquote (“)': np.mean([t.count('“') / (len(t.split())+1) * 1000 for t in texts]),
        'curly_squote (’ )': np.mean([t.count('’') / (len(t.split())+1) * 1000 for t in texts]),
        'straight_dquote (")': np.mean([t.count('"') / (len(t.split())+1) * 1000 for t in texts]),
        'parens (()': np.mean([t.count('(') / (len(t.split())+1) * 1000 for t in texts]),
        'asterisk (*)': np.mean([t.count('*') / (len(t.split())+1) * 1000 for t in texts]),
        'exclamation (!)': np.mean([t.count('!') / (len(t.split())+1) * 1000 for t in texts]),
    }
print(pd.DataFrame(gpt_feat).to_string())

