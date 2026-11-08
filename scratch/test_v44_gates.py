import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

K = 18
LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])

idx_astra = LABELS.index('gpt-6-astra')
idx_terra = LABELS.index('gpt-5.6-terra')
idx_luna = LABELS.index('gpt-5.6-luna')
idx_sol = LABELS.index('gpt-5.6-sol')

print('Astra idx:', idx_astra)
print('Terra idx:', idx_terra)

# Test Astra gate on synthetic text
txt_low = 'This is an answer with very few commas and clean sentences that explain the concept clearly.'
words = len(txt_low.split())
comma_rate = txt_low.count(',') / words * 100
print(f'Low text words: {words}, comma rate: {comma_rate:.2f}')

P = np.ones((1, K)) / K
if words >= 10:
    if comma_rate < 7.5:
        if P[0, idx_astra] > 0.04:
            P[0, idx_astra] *= 1.12
            P[0, idx_terra] *= 0.92
            P[0, idx_luna] *= 0.92

P /= P.sum(1, keepdims=True)
print(f'P[astra] after boost: {P[0, idx_astra]:.4f}, P[terra]: {P[0, idx_terra]:.4f}')
assert P[0, idx_astra] > P[0, idx_terra], 'Astra should be higher than Terra!'
print('TEST PASSED!')

