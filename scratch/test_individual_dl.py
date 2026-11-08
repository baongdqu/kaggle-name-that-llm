import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score

train = pd.read_csv('data/train.csv')
p1 = r'scratch\out_tokattn1\tokattn_oof.npz'
p2 = r'scratch\out_tokattn2\tokattn_q8l22_oof.npz'
p3 = r'scratch\out_tokattn2\tokattn_debl23_oof.npz'

d1 = np.load(p1, allow_pickle=True)
d2 = np.load(p2, allow_pickle=True)
d3 = np.load(p3, allow_pickle=True)

LABELS = ['claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
 'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash','glm-5.3','glm-5.3-flash',
 'gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra','kimi-k3','qwen3.8-max']
L2I = {l: i for i, l in enumerate(LABELS)}
y_true = np.array([L2I[m] for m in train['model'].values])

idx_map1 = [list(d1['classes']).index(l) for l in LABELS]
idx_map2 = [list(d2['classes']).index(l) for l in LABELS]
idx_map3 = [list(d3['classes']).index(l) for l in LABELS]

m1 = d1['probs'][:, idx_map1]
m2 = d2['probs'][:, idx_map2]
m3 = d3['probs'][:, idx_map3]

all_pids = train['prompt_id'].values
unique_pids = np.unique(all_pids)

def score_hungarian(probs):
    preds = np.zeros(len(train), dtype=int)
    log_p = np.log(probs + 1e-12)
    for p in unique_pids:
        idx = np.where(all_pids == p)[0]
        cost = -log_p[idx]
        r, c = linear_sum_assignment(cost)
        preds[idx[r]] = c
    return balanced_accuracy_score(y_true, preds)

print(f"Model 1 (Qwen 14B):        {score_hungarian(m1)*100:.2f}%")
print(f"Model 2 (Qwen 8B):         {score_hungarian(m2)*100:.2f}%")
print(f"Model 3 (DeBERTa Large):   {score_hungarian(m3)*100:.2f}%")
print(f"Blend (Equal 1:1:1):       {score_hungarian((m1+m2+m3)/3)*100:.2f}%")

# Grid search for best weights among the 3 DL models:
best_w_acc = 0.0
best_w = None
for w1 in [0.0, 0.5, 1.0, 1.5, 2.0]:
    for w2 in [0.0, 0.5, 1.0, 1.5]:
        for w3 in [0.0, 0.5, 1.0, 1.5]:
            if w1 + w2 + w3 == 0: continue
            blend = w1 * m1 + w2 * m2 + w3 * m3
            acc = score_hungarian(blend)
            if acc > best_w_acc:
                best_w_acc = acc
                best_w = (w1, w2, w3)

print(f"🏆 Best DL Blend: {best_w_acc*100:.2f}% with weights {best_w}")

