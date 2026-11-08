import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

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

dl_blend = (d1['probs'][:, idx_map1] + d2['probs'][:, idx_map2] + d3['probs'][:, idx_map3]) / 3.0
log_dl = np.log(dl_blend + 1e-12)

hung_preds = dl_blend.argmax(axis=1).copy()
for pid in train['prompt_id'].unique():
    idx = np.where(train['prompt_id'] == pid)[0]
    r, c = linear_sum_assignment(-log_dl[idx])
    hung_preds[idx[r]] = c

cm = confusion_matrix(y_true, hung_preds)
print("=== Top Confused Pairs in Baseline Deep Learning + Hungarian ===")
confusions = []
for i in range(18):
    for j in range(18):
        if i != j and cm[i, j] > 0:
            confusions.append((cm[i, j], LABELS[i], LABELS[j]))

confusions.sort(key=lambda x: x[0], reverse=True)
for count, true_cls, pred_cls in confusions[:20]:
    print(f"  {true_cls:24s} -> mistaken as -> {pred_cls:24s}: {count:2d} times")

