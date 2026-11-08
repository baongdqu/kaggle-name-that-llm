import numpy as np
import pandas as pd
import re

train = pd.read_csv('data/train.csv')

def analyze_family(models, name):
    print(f"\n=================== {name} FAMILY ANALYSIS ===================")
    sub = train[train.model.isin(models)].copy()
    sub['word_count'] = sub.text.apply(lambda t: len(str(t).split()))
    sub['char_count'] = sub.text.apply(lambda t: len(str(t)))
    sub['avg_word_len'] = sub.char_count / np.maximum(sub.word_count, 1)
    sub['num_newlines'] = sub.text.apply(lambda t: str(t).count('\n'))
    sub['num_digits'] = sub.text.apply(lambda t: sum(c.isdigit() for c in str(t)))
    sub['has_code'] = sub.text.apply(lambda t: int('```' in str(t)))
    sub['num_colons'] = sub.text.apply(lambda t: str(t).count(':'))
    sub['num_bullets'] = sub.text.apply(lambda t: len(re.findall(r'^\s*[-*•]\s', str(t), re.MULTILINE)))
    sub['ttr'] = sub.text.apply(lambda t: len(set(str(t).lower().split())) / max(1, len(str(t).split())))
    
    stats = sub.groupby('model')[['word_count', 'avg_word_len', 'num_newlines', 'num_bullets', 'ttr']].mean()
    print(stats.to_string())

analyze_family(['gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'], 'GEMINI')
analyze_family(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra'], 'GPT')
analyze_family(['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5'], 'CLAUDE')
analyze_family(['glm-5.3', 'glm-5.3-flash'], 'GLM')

