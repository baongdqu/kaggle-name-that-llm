import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

train = pd.read_csv('data/train.csv')
labels = sorted(train.model.unique())
l2i = {l: i for i, l in enumerate(labels)}
y_true = np.array([l2i[m] for m in train.model.values])
K = 18

cache = np.load('lab/cache/base_oof.npz')
base_probs = cache['probs']
base_logits = np.log(np.maximum(base_probs, 1e-12))
prompts = train.prompt_id.values

# Calculate Prompt-Relative Features for every passage
sub = train.copy()
sub['wlen'] = sub.text.apply(lambda t: len(str(t).split()))
sub['char_len'] = sub.text.apply(lambda t: len(str(t)))
sub['avg_wlen'] = sub.char_len / np.maximum(sub['wlen'], 1)
sub['ttr'] = sub.text.apply(lambda t: len(set(str(t).lower().split())) / max(1, len(str(t).split())))

# Compute relative z-scores per prompt cluster
z_len = np.zeros(len(train), dtype=np.float32)
z_ttr = np.zeros(len(train), dtype=np.float32)
z_avg_wlen = np.zeros(len(train), dtype=np.float32)

for p in np.unique(prompts):
    idx = np.where(prompts == p)[0]
    if len(idx) > 1:
        # z-score length
        l_ = sub.loc[idx, 'wlen'].values
        std_l = l_.std()
        if std_l > 1e-3: z_len[idx] = (l_ - l_.mean()) / std_l
        
        # z-score ttr
        t_ = sub.loc[idx, 'ttr'].values
        std_t = t_.std()
        if std_t > 1e-3: z_ttr[idx] = (t_ - t_.mean()) / std_t
        
        # z-score avg word length
        w_ = sub.loc[idx, 'avg_wlen'].values
        std_w = w_.std()
        if std_w > 1e-3: z_avg_wlen[idx] = (w_ - w_.mean()) / std_w

# Measure model-specific mean relative bias
mean_z_len = np.array([z_len[y_true == i].mean() for i in range(K)])
mean_z_ttr = np.array([z_ttr[y_true == i].mean() for i in range(K)])
mean_z_avg_wlen = np.array([z_avg_wlen[y_true == i].mean() for i in range(K)])

print("=== PROMPT-RELATIVE FEATURE BIASES PER CLASS ===")
for i in range(K):
    print(f"{labels[i]:22s} | z_len: {mean_z_len[i]:+6.3f} | z_ttr: {mean_z_ttr[i]:+6.3f} | z_avg_wlen: {mean_z_avg_wlen[i]:+6.3f}")

# Baseline free score
score_base = balanced_accuracy_score(y_true, base_logits.argmax(axis=1))
print(f"\nBase OOF Score: {score_base:.4f}")

# Test adding multi-feature relative bias
for w_l in [0.0, 0.05, 0.10, 0.15]:
    for w_t in [0.0, 0.02, 0.05, 0.08]:
        for w_w in [0.0, 0.02, 0.05, 0.08]:
            mod_logits = base_logits.copy()
            for i in range(K):
                mod_logits[:, i] += w_l * z_len * mean_z_len[i] + w_t * z_ttr * mean_z_ttr[i] + w_w * z_avg_wlen * mean_z_avg_wlen[i]
            sc = balanced_accuracy_score(y_true, mod_logits.argmax(axis=1))
            if sc > score_base + 0.003:
                print(f"w_len={w_l:.2f}, w_ttr={w_t:.2f}, w_avg={w_w:.2f} -> Score: {sc:.4f} (Gain: {sc - score_base:+.4f})")

