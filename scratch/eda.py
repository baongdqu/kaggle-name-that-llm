import pandas as pd
import numpy as np

def eda():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    print("=== TRAIN DATA SHAPE ===")
    print(train.shape)
    print("\n=== TRAIN COLUMNS ===")
    print(train.columns.tolist())
    print("\n=== CLASS DISTRIBUTION (18 MODELS) ===")
    print(train['model'].value_counts())
    print(f"\nUnique models count: {train['model'].nunique()}")
    print(f"Unique prompt_ids count: {train['prompt_id'].nunique()}")

    print("\n=== SAMPLES PER PROMPT ===")
    prompt_counts = train.groupby('prompt_id')['model'].nunique()
    print("Does every prompt have all 18 models?")
    print(prompt_counts.value_counts())

    train['word_count'] = train['text'].apply(lambda x: len(str(x).split()))
    train['char_count'] = train['text'].apply(lambda x: len(str(x)))

    print("\n=== WORD COUNT STATS BY MODEL ===")
    word_stats = train.groupby('model')['word_count'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(word_stats.sort_values(by='mean', ascending=False))

    print("\n=== TEST DATA ===")
    print(f"Test shape: {test.shape}")
    print(test.head())

if __name__ == '__main__':
    eda()
