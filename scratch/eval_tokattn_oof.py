import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

train = pd.read_csv('data/train.csv')
labels = sorted(train.model.unique())
y_true = train.model.map({l: i for i, l in enumerate(labels)}).values

tok_deb = np.load('scratch/out_tokattn2/tokattn_debl23_oof.npz')
tok_q8 = np.load('scratch/out_tokattn2/tokattn_q8l22_oof.npz')

p_deb = tok_deb['probs']
p_q8 = tok_q8['probs']

acc_deb = balanced_accuracy_score(y_true, p_deb.argmax(axis=1))
acc_q8 = balanced_accuracy_score(y_true, p_q8.argmax(axis=1))

print(f"TokAttn DeBERTa-v3-Large L23 OOF Balanced Accuracy: {acc_deb:.4f}")
print(f"TokAttn Qwen3-8B-Base L22 OOF Balanced Accuracy:     {acc_q8:.4f}")

# Check ensemble of both
p_blend = 0.5 * p_deb + 0.5 * p_q8
acc_blend = balanced_accuracy_score(y_true, p_blend.argmax(axis=1))
print(f"Blend (Deb + Q8) OOF Balanced Accuracy:             {acc_blend:.4f}")

