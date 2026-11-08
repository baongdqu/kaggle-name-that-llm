import pandas as pd

df = pd.read_csv('data/train.csv')
print('=== All models using spaced em-dash in train.csv ===')
print(df[df['text'].str.contains(' \u2014 ')]['model'].value_counts())

print('\n=== All models using curly double quotes \u201c ===')
print(df[df['text'].str.contains('\u201c')]['model'].value_counts())

print('\n=== All models using curly single quotes \u2019 ===')
print(df[df['text'].str.contains('\u2019')]['model'].value_counts())
