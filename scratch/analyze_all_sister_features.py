import pandas as pd, numpy as np, re
from collections import Counter

train = pd.read_csv('data/train.csv')
print("Loaded train.csv:", len(train))

def analyze_pair(m1, m2):
    print("\n" + "="*70)
    print(f"🔬 FORENSIC COMPARISON: {m1} vs {m2}")
    print("="*70)
    
    sub1 = train[train['model'] == m1].sort_values('prompt_id').reset_index(drop=True)
    sub2 = train[train['model'] == m2].sort_values('prompt_id').reset_index(drop=True)
    
    # 1. Word counts & relative lengths
    w1 = np.array([len(t.split()) for t in sub1['text'].values])
    w2 = np.array([len(t.split()) for t in sub2['text'].values])
    diff_w = w1 - w2
    print(f"📊 Word count: {m1} mean={w1.mean():.1f} (std={w1.std():.1f}) | {m2} mean={w2.mean():.1f} (std={w2.std():.1f})")
    print(f"   In {(diff_w > 0).mean()*100:.1f}% of prompts, {m1} is longer than {m2} (mean diff: {diff_w.mean():+.1f} words)")
    
    # 2. Punctuation frequencies
    puncs = [':', ';', '—', '–', '-', '“', '”', '’', '‘', '(', ')', '/', '!', '?', '`', '*']
    print("\n📌 Punctuation counts per 1000 words:")
    for p in puncs:
        c1 = sum(t.count(p) for t in sub1['text'].values) / (w1.sum() / 1000)
        c2 = sum(t.count(p) for t in sub2['text'].values) / (w2.sum() / 1000)
        ratio = (c1 + 1e-4) / (c2 + 1e-4)
        if ratio > 1.3 or ratio < 0.7:
            print(f"   '{p}': {m1}={c1:.2f} vs {m2}={c2:.2f} (ratio: {ratio:.2f}) ⚡")
            
    # 3. Sentence counts & average sentence length
    s1 = np.array([len(re.findall(r'[.!?]+', t)) for t in sub1['text'].values])
    s2 = np.array([len(re.findall(r'[.!?]+', t)) for t in sub2['text'].values])
    print(f"\n📝 Sentences: {m1} mean={s1.mean():.1f} sent, {w1.mean()/(s1.mean()+1e-5):.1f} w/s | {m2} mean={s2.mean():.1f} sent, {w2.mean()/(s2.mean()+1e-5):.1f} w/s")
    
    # 4. Starting words
    start1 = [t.split()[0].lower() for t in sub1['text'].values if t.split()]
    start2 = [t.split()[0].lower() for t in sub2['text'].values if t.split()]
    top_s1 = Counter(start1).most_common(5)
    top_s2 = Counter(start2).most_common(5)
    print(f"\n🚀 Top opening words:\n   {m1}: {top_s1}\n   {m2}: {top_s2}")
    
    # 5. Top distinctive word n-grams (TF-IDF differential)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X = vec.fit_transform(list(sub1['text'].values) + list(sub2['text'].values))
    vocab = np.array(vec.get_feature_names_out())
    mean1 = np.array(X[:len(sub1)].mean(axis=0)).flatten()
    mean2 = np.array(X[len(sub1):].mean(axis=0)).flatten()
    diff = mean1 - mean2
    
    top_m1 = vocab[np.argsort(diff)[-10:][::-1]]
    top_m2 = vocab[np.argsort(diff)[:10]]
    print(f"\n✨ Distinctive for {m1}: {list(top_m1)}")
    print(f"✨ Distinctive for {m2}: {list(top_m2)}")

analyze_pair('gemini-3.7-flash', 'gemini-3.8-flash')
analyze_pair('gpt-5.6-sol', 'gpt-5.6-terra')
analyze_pair('glm-5.3', 'glm-5.3-flash')
analyze_pair('claude-fable-5', 'glm-5.3-flash')

