import numpy as np, pandas as pd, re
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv('data/train.csv')

def extract_advanced_forensics(texts):
    feats = []
    for t in texts:
        t = str(t)
        n_char = max(len(t), 1)
        lines = t.splitlines()
        n_lines = max(len(lines), 1)
        words = re.findall(r"[A-Za-z]+(?:[’'][A-Za-z]+)?", t)
        n_words = max(len(words), 1)
        
        # 1. Punctuation frequencies
        c_excl = t.count('!') / n_words
        c_quest = t.count('?') / n_words
        c_semi = t.count(';') / n_words
        c_colon = t.count(':') / n_words
        c_comma = t.count(',') / n_words
        c_dash_en = t.count('–') / n_words
        c_dash_em = t.count('—') / n_words
        c_hyphen = t.count('-') / n_words
        c_quotes_curly = (t.count('“') + t.count('”')) / n_words
        c_quotes_straight = t.count('"') / n_words
        c_apostrophe = (t.count("'") + t.count("’")) / n_words
        c_ellipsis = (t.count('…') + t.count('...')) / n_words
        c_parens = (t.count('(') + t.count(')')) / n_words
        c_brackets = (t.count('[') + t.count(']')) / n_words
        c_asterisk = t.count('*') / n_words
        c_backtick = t.count('`') / n_words
        
        # 2. Markdown and Structural formatting
        has_code_block = 1.0 if '```' in t else 0.0
        bullet_dash = sum(1 for l in lines if l.strip().startswith('- ')) / n_lines
        bullet_star = sum(1 for l in lines if l.strip().startswith('* ')) / n_lines
        bullet_num = sum(1 for l in lines if re.match(r'^\d+\.', l.strip())) / n_lines
        headers = sum(1 for l in lines if l.strip().startswith('#')) / n_lines
        empty_lines = sum(1 for l in lines if not l.strip()) / n_lines
        
        # 3. Capitalization & Case patterns
        colon_caps = len(re.findall(r':\s+[A-Z]', t)) / max(t.count(':'), 1)
        upper_ratio = sum(c.isupper() for c in t) / n_char
        digit_ratio = sum(c.isdigit() for c in t) / n_char
        
        # 4. Word length & Lexical diversity
        w_lens = [len(w) for w in words] or [0]
        avg_w_len = float(np.mean(w_lens))
        std_w_len = float(np.std(w_lens))
        ttr = len(set(w.lower() for w in words)) / n_words
        long_words = sum(1 for w in words if len(w) >= 7) / n_words
        
        # 5. Contractions vs Formal Full Words
        n_dont = len(re.findall(r"\b(?:don't|don’t)\b", t, re.I)) / n_words
        n_do_not = len(re.findall(r"\bdo not\b", t, re.I)) / n_words
        n_cant = len(re.findall(r"\b(?:can't|can’t)\b", t, re.I)) / n_words
        n_cannot = len(re.findall(r"\bcannot\b", t, re.I)) / n_words
        n_its = len(re.findall(r"\b(?:it's|it’t)\b", t, re.I)) / n_words
        n_it_is = len(re.findall(r"\bit is\b", t, re.I)) / n_words
        
        # 6. Specific discourse markers
        n_however = len(re.findall(r"\bhowever\b", t, re.I)) / n_words
        n_furthermore = len(re.findall(r"\bfurthermore\b", t, re.I)) / n_words
        n_moreover = len(re.findall(r"\bmoreover\b", t, re.I)) / n_words
        n_therefore = len(re.findall(r"\btherefore\b", t, re.I)) / n_words
        n_overall = len(re.findall(r"\boverall\b", t, re.I)) / n_words
        n_specifically = len(re.findall(r"\bspecifically\b", t, re.I)) / n_words
        n_additionally = len(re.findall(r"\badditionally\b", t, re.I)) / n_words
        
        row = [
            np.log1p(n_words), np.log1p(n_char), n_char / n_words, n_lines / n_words,
            c_excl, c_quest, c_semi, c_colon, c_comma, c_dash_en, c_dash_em, c_hyphen,
            c_quotes_curly, c_quotes_straight, c_apostrophe, c_ellipsis, c_parens, c_brackets,
            c_asterisk, c_backtick, has_code_block, bullet_dash, bullet_star, bullet_num,
            headers, empty_lines, colon_caps, upper_ratio, digit_ratio, avg_w_len, std_w_len,
            ttr, long_words, n_dont, n_do_not, n_cant, n_cannot, n_its, n_it_is,
            n_however, n_furthermore, n_moreover, n_therefore, n_overall, n_specifically, n_additionally
        ]
        feats.append(row)
    return np.array(feats, dtype=np.float32)

print("Extracting 46 Forensic Descriptors...", flush=True)
X_raw = extract_advanced_forensics(train['text'])

# Also compute prompt-relative Z-scores of all 46 features:
pids = train['prompt_id'].values
unique_pids = np.unique(pids)
X_z = np.zeros_like(X_raw)
for p in unique_pids:
    idx = np.where(pids == p)[0]
    sub = X_raw[idx]
    m = sub.mean(axis=0)
    s = np.maximum(sub.std(axis=0), 1e-4)
    X_z[idx] = (sub - m) / s

X_forensic_all = np.column_stack([X_raw, X_z])
print(f"Total Forensic Matrix: {X_forensic_all.shape}")

gkf = GroupKFold(n_splits=5)

def benchmark_pair(m1, m2):
    sub_mask = train['model'].isin([m1, m2])
    sub_idx = np.where(sub_mask)[0]
    y_sub = (train['model'].values[sub_idx] == m2).astype(int)
    p_sub = pids[sub_idx]
    X_sub = X_forensic_all[sub_idx]
    
    print(f"\n🔬 Evaluating Pair [{m1}] vs [{m2}] (N={len(sub_idx)}):")
    for name, clf in [
        ('Ridge (alpha=10.0)', make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0))),
        ('Logistic (C=0.05)', make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500))),
        ('Logistic (C=0.2)', make_pipeline(StandardScaler(), LogisticRegression(C=0.2, max_iter=500))),
        ('ExtraTrees (100)', ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=42)),
    ]:
        oof = np.zeros(len(sub_idx))
        for tr, va in gkf.split(X_sub, y_sub, groups=p_sub):
            clf.fit(X_sub[tr], y_sub[tr])
            if hasattr(clf, 'predict_proba'):
                oof[va] = clf.predict_proba(X_sub[va])[:, 1]
            elif hasattr(clf, 'decision_function'):
                oof[va] = clf.decision_function(X_sub[va])
            else:
                oof[va] = clf.predict(X_sub[va])
        
        # Evaluate accuracy
        if hasattr(clf, 'decision_function'):
            preds = (oof > 0).astype(int)
        else:
            preds = (oof > 0.5).astype(int)
        acc = accuracy_score(y_sub, preds)
        print(f"  {name:25s}: CV Acc = {acc*100:.2f}%")

benchmark_pair('glm-5.3', 'glm-5.3-flash')
benchmark_pair('gemini-3.7-flash', 'gemini-3.8-flash')
benchmark_pair('gpt-5.6-sol', 'gpt-5.6-terra')
benchmark_pair('gpt-5.6-luna', 'gpt-5.6-sol')

