import pandas as pd

train = pd.read_csv('data/train.csv')
labels_train = sorted(train.model.unique())

infer_labels = ['claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
 'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash','glm-5.3','glm-5.3-flash',
 'gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra','kimi-k3','qwen3.8-max']

print("Match:", labels_train == infer_labels)
if labels_train != infer_labels:
    for i, (a, b) in enumerate(zip(labels_train, infer_labels)):
        print(f"{i}: train={a} vs infer={b}")

