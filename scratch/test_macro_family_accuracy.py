import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score

train = pd.read_csv('data/train.csv')
p1 = r'scratch\out_tokattn1\tokattn_oof.npz'
p2 = r'scratch\out_tokattn2\tokattn_q8l22_oof.npz'
p3 = r'scratch\out_tokattn2\tokattn_debl23_oof.npz'

d1 = np.load(p1, allow_pickle=True)
d2 = np.load(p2, allow_pickle=True)
d3 = np.load(p3, allow_pickle=True)

LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
                 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
                 'claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'])
L2I = {l: i for i, l in enumerate(LABELS)}
y_true = np.array([L2I[m] for m in train['model'].values])

idx_map1 = [list(d1['classes']).index(l) for l in LABELS]
idx_map2 = [list(d2['classes']).index(l) for l in LABELS]
idx_map3 = [list(d3['classes']).index(l) for l in LABELS]

dl_blend = (1.0 * d1['probs'][:, idx_map1] + 1.0 * d2['probs'][:, idx_map2] + 1.5 * d3['probs'][:, idx_map3]) / 3.5

# Group into 6 macro-families:
# 0: gpt (5.5, 5.6-luna, 5.6-sol, 5.6-terra, 6-astra)
# 1: gemini (3.1-pro, 3.6-flash, 3.7-flash, 3.8-flash)
# 2: claude (fable-5, opus-4-6, opus-4-8, opus-5, sonnet-5)
# 3: glm (5.3, 5.3-flash)
# 4: qwen (qwen3.8-max)
# 5: kimi (kimi-k3)

family_map = {}
for m in LABELS:
    if m.startswith('gpt'): family_map[L2I[m]] = 0
    elif m.startswith('gemini'): family_map[L2I[m]] = 1
    elif m.startswith('claude'): family_map[L2I[m]] = 2
    elif m.startswith('glm'): family_map[L2I[m]] = 3
    elif m.startswith('qwen'): family_map[L2I[m]] = 4
    elif m.startswith('kimi'): family_map[L2I[m]] = 5

y_fam_true = np.array([family_map[y] for y in y_true])

fam_probs = np.zeros((len(train), 6))
for i in range(18):
    fam_probs[:, family_map[i]] += dl_blend[:, i]

y_fam_pred = np.argmax(fam_probs, axis=1)
acc = balanced_accuracy_score(y_fam_true, y_fam_pred)

print(f"🏛️ Deep Learning Macro-Family Balanced Accuracy: {acc*100:.2f}%")
for f_idx, f_name in enumerate(['GPT', 'Gemini', 'Claude', 'GLM', 'Qwen', 'Kimi']):
    mask = (y_fam_true == f_idx)
    sub_acc = (y_fam_pred[mask] == f_idx).mean()
    print(f"   - {f_name:8s}: {sub_acc*100:.2f}% accuracy ({mask.sum()} samples)")
