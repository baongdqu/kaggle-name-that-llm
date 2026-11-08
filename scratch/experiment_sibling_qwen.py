import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

df = pd.read_csv('data/train.csv')
gemini_mask = df['model'].str.contains('gemini-3.[678]-flash').values

# Load valid features from fold 0
vf = np.load('dataset_sibling_lora/valid_features.npz', allow_pickle=True)
print('Keys in valid_features:', list(vf.keys()))
