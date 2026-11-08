import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv')
print(f"Loaded train: {len(train)} rows")

# 1. Claude Family: claude-fable-5, claude-opus-4-6, claude-opus-4-8, claude-opus-5, claude-sonnet-5
claude_df = train[train.model.str.startswith('claude')].copy()
claude_df['word_count'] = claude_df.text.apply(lambda x: len(x.split()))
claude_df['char_count'] = claude_df.text.apply(len)
claude_df['sentence_count'] = claude_df.text.apply(lambda x: x.count('.') + x.count('!') + x.count('?'))

print("\n=== CLAUDE FAMILY STATS ===")
for m in sorted(claude_df.model.unique()):
    sub = claude_df[claude_df.model == m]
    print(f"{m:20s}: words={sub.word_count.mean():.1f} (std={sub.word_count.std():.1f}), chars={sub.char_count.mean():.1f}, sents={sub.sentence_count.mean():.1f}")

# 2. GPT Family: gpt-5.5, gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-6-astra
gpt_df = train[train.model.str.startswith('gpt')].copy()
gpt_df['word_count'] = gpt_df.text.apply(lambda x: len(x.split()))
gpt_df['colon_count'] = gpt_df.text.apply(lambda x: x.count(':'))
gpt_df['semicolon_count'] = gpt_df.text.apply(lambda x: x.count(';'))

print("\n=== GPT FAMILY STATS ===")
for m in sorted(gpt_df.model.unique()):
    sub = gpt_df[gpt_df.model == m]
    print(f"{m:20s}: words={sub.word_count.mean():.1f}, colons={sub.colon_count.mean():.2f}, semicolons={sub.semicolon_count.mean():.2f}")

# 3. Gemini Family: gemini-3.1-pro-preview, gemini-3.6-flash, gemini-3.7-flash, gemini-3.8-flash
gemini_df = train[train.model.str.startswith('gemini')].copy()
gemini_df['word_count'] = gemini_df.text.apply(lambda x: len(x.split()))
gemini_df['bullet_count'] = gemini_df.text.apply(lambda x: x.count('- ') + x.count('* '))
gemini_df['number_list'] = gemini_df.text.apply(lambda x: x.count('1. ') + x.count('2. ') + x.count('3. '))

print("\n=== GEMINI FAMILY STATS ===")
for m in sorted(gemini_df.model.unique()):
    sub = gemini_df[gemini_df.model == m]
    print(f"{m:25s}: words={sub.word_count.mean():.1f}, bullets={sub.bullet_count.mean():.2f}, num_lists={sub.number_list.mean():.2f}")
