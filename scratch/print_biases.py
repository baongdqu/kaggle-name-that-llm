import numpy as np
import pandas as pd

train = pd.read_csv('data/train.csv')
labels = sorted(train.model.unique())
l2i = {l: i for i, l in enumerate(labels)}
y_true = np.array([l2i[m] for m in train.model.values])
prompts = train.prompt_id.values

wlen = np.array([len(str(t).split()) for t in train.text.values], dtype=np.float32)
char_len = np.array([len(str(t)) for t in train.text.values], dtype=np.float32)
avg_wlen = char_len / np.maximum(wlen, 1.0)

z_l = np.zeros(len(train), dtype=np.float32)
z_w = np.zeros(len(train), dtype=np.float32)

for p in np.unique(prompts):
    idx = np.where(prompts == p)[0]
    if len(idx) >= 3:
        l_ = wlen[idx]; s_l = l_.std()
        if s_l > 1.0: z_l[idx] = (l_ - l_.mean()) / s_l
    if len(idx) >= 6:
        w_ = avg_wlen[idx]; s_w = w_.std()
        if s_w > 0.05: z_w[idx] = (w_ - w_.mean()) / s_w

b_l = np.array([z_l[y_true == i].mean() for i in range(18)], dtype=np.float32)
b_w = np.array([z_w[y_true == i].mean() for i in range(18)], dtype=np.float32)

print("=== BIAS VECTORS (18 classes) ===")
print("LABELS:", labels)
print("b_l (doc length bias):", [round(float(x), 4) for x in b_l])
print("b_w (avg word length bias):", [round(float(x), 4) for x in b_w])

for i, l in enumerate(labels):
    print(f"{i:2d} | {l:22s} | b_l={b_l[i]:+6.3f} | b_w={b_w[i]:+6.3f}")

