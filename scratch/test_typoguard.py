import pandas as pd
import numpy as np

df = pd.read_csv('data/train.csv')
LABELS = sorted(df['model'].unique())

print("Testing TypoGuard precision on train.csv (2,699 samples):")

# Test Rule 1: '’' in text -> NOT Claude, NOT GLM
claude_glm = [m for m in LABELS if 'claude' in m or 'glm' in m]
samples_with_curly = df[df['text'].str.contains('’')]
print(f"Total samples with curly apostrophe ’: {len(samples_with_curly)}")
violations = samples_with_curly[samples_with_curly['model'].isin(claude_glm)]
print(f"Violations (any Claude or GLM with curly apos?): {len(violations)}")
assert len(violations) == 0, "Violation found!"
print("Rule 1 100% ACCURATE!")

# Test Rule 2: '—' in text -> NOT Gemini
gemini = [m for m in LABELS if 'gemini' in m]
samples_with_emdash = df[df['text'].str.contains('—')]
print(f"Total samples with em-dash —: {len(samples_with_emdash)}")
gemini_emdash = samples_with_emdash[samples_with_emdash['model'].isin(gemini)]
print(f"Gemini with em-dash: {len(gemini_emdash)} out of {len(samples_with_emdash)} (only {len(gemini_emdash)/len(df[df['model'].isin(gemini)])*100:.2f}% of all Gemini)")
