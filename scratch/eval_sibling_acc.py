import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score

df = pd.read_csv('data/train.csv')
id2label = dict(zip(df['id'], df['model']))

for fold in range(5):
    vf = np.load(f'scratch/lora24_sibling_out/lora_out/job_{fold}/valid_features.npz', allow_pickle=True)
    rf = np.load(f'scratch/lora24_sibling_out/lora_out/job_{fold}/readouts.npz')
    ids = vf['ids']
    X = vf['features']
    y_true = np.array([id2label[i] for i in ids])
    
    classes = rf['classes']
    logits = X @ rf['plain_coef'].T + rf['plain_intercept']
    y_pred = classes[np.argmax(logits, axis=1)]
    
    # Check overall
    overall_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Check Gemini Flash subset
    g_mask = np.isin(y_true, ['gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash'])
    g_acc = balanced_accuracy_score(y_true[g_mask], y_pred[g_mask])
    
    # Check GPT-5 subset
    gpt_mask = np.isin(y_true, ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra'])
    gpt_acc = balanced_accuracy_score(y_true[gpt_mask], y_pred[gpt_mask])
    
    print(f'Fold {fold}: Overall = {overall_acc:.4f} | Gemini Flash = {g_acc:.4f} | GPT-5 = {gpt_acc:.4f}')
