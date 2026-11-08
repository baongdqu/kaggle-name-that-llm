import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
import scipy.sparse as sp

# 1. Load Data
train = pd.read_csv('data/train.csv')
LABELS = sorted(train.model.unique())
L2I = {l: i for i, l in enumerate(LABELS)}
y_true = np.array([L2I[m] for m in train.model.values])
prompts = train.prompt_id.values

cache = np.load('lab/cache/base_oof.npz')
base_probs = cache['probs'].copy()

# 2. Define the 3 Specialist Groups
TRIO_GEMINI = [L2I['gemini-3.6-flash'], L2I['gemini-3.7-flash'], L2I['gemini-3.8-flash']]
TRIO_GPT = [L2I['gpt-5.6-luna'], L2I['gpt-5.6-sol'], L2I['gpt-5.6-terra']]
DUO_GLM = [L2I['glm-5.3'], L2I['glm-5.3-flash']]

SPECIALIST_GROUPS = {
    'gemini_flash': TRIO_GEMINI,
    'gpt56': TRIO_GPT,
    'glm53': DUO_GLM
}

print("=== SPECIALIST TARGET GROUPS ===")
for name, group in SPECIALIST_GROUPS.items():
    print(f"  {name:15s}: {[LABELS[i] for i in group]} (indices: {group})")

# 3. Extract Stylometric & Function Word Features
FUNCTION_WORDS = [
    'the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'for', 'it', 'with', 'as', 'was', 'on', 'at', 'by', 'this',
    'be', 'are', 'from', 'or', 'an', 'they', 'which', 'one', 'you', 'were', 'all', 'there', 'would', 'their', 'we',
    'been', 'has', 'when', 'who', 'will', 'more', 'no', 'if', 'out', 'so', 'what', 'up', 'its', 'about', 'into',
    'than', 'them', 'can', 'only', 'other', 'some', 'could', 'time', 'these', 'two', 'may', 'then', 'do', 'first',
    'any', 'now', 'such', 'like', 'over', 'even', 'most', 'also', 'after', 'did', 'many', 'before', 'must', 'through',
    'where', 'much', 'should', 'because', 'each', 'just', 'those', 'how', 'too', 'very', 'still', 'between', 'both'
]

def extract_micro_features(texts):
    vec_fw = TfidfVectorizer(vocabulary=FUNCTION_WORDS, lowercase=True, norm='l2', sublinear_tf=True)
    X_fw = vec_fw.fit_transform(texts)
    
    vec_char = TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', min_df=3, max_features=15000, sublinear_tf=True)
    X_char = vec_char.fit_transform(texts)
    
    # Statistical sentence and punctuation features
    stats = []
    for t in texts:
        s_text = str(t)
        words = s_text.split()
        n_words = max(1, len(words))
        sentences = [s.strip() for s in re.split(r'[.!?]+', s_text) if s.strip()]
        n_sents = max(1, len(sentences))
        sent_lens = [len(s.split()) for s in sentences]
        
        # Features
        mean_sent_len = np.mean(sent_lens)
        std_sent_len = np.std(sent_lens) if len(sent_lens) > 1 else 0.0
        skew_sent_len = float(np.mean(((sent_lens - mean_sent_len) / max(std_sent_len, 1e-4)) ** 3)) if std_sent_len > 1e-4 else 0.0
        
        comma_rate = s_text.count(',') / n_words
        semicolon_rate = s_text.count(';') / n_words
        colon_rate = s_text.count(':') / n_words
        emdash_rate = (s_text.count('—') + s_text.count('–')) / n_words
        paren_rate = s_text.count('(') / n_words
        quote_rate = (s_text.count('"') + s_text.count("'") + s_text.count('“') + s_text.count('”')) / n_words
        hyphen_rate = s_text.count('-') / n_words
        avg_wlen = len(s_text) / n_words
        
        stats.append([
            mean_sent_len, std_sent_len, skew_sent_len,
            comma_rate, semicolon_rate, colon_rate, emdash_rate,
            paren_rate, quote_rate, hyphen_rate, avg_wlen
        ])
    
    X_stats = np.array(stats, dtype=np.float32)
    # Z-score normalize stats
    X_stats = (X_stats - X_stats.mean(axis=0)) / (X_stats.std(axis=0) + 1e-6)
    
    X_full = sp.hstack([X_fw, X_char, sp.csr_matrix(X_stats)]).tocsr()
    return X_full, vec_fw, vec_char

print("\nExtracting micro-stylometrics across all 2,699 training samples...")
X_full, vec_fw, vec_char = extract_micro_features(train.text.values)
print(f"Extracted feature matrix shape: {X_full.shape}")

# 4. 5-Fold GroupKFold Cross-Validation on the 3 Specialist Groups
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(train, y_true, groups=prompts))

oof_specialist_probs = {name: np.zeros((len(train), len(group)), dtype=np.float64) for name, group in SPECIALIST_GROUPS.items()}

print("\n=== EVALUATING 5-FOLD GROUPKFOLD SPECIALIST HEADS ===")
for name, group in SPECIALIST_GROUPS.items():
    group_mask = np.isin(y_true, group)
    y_group = np.array([group.index(y) if y in group else -1 for y in y_true])
    
    acc_baseline = []
    acc_specialist = []
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        # Subset to group members in training fold
        tr_mask = train_idx[np.isin(y_true[train_idx], group)]
        va_mask = val_idx[np.isin(y_true[val_idx], group)]
        
        if len(va_mask) == 0:
            continue
            
        # Baseline accuracy on this fold within this group
        base_pred = base_probs[va_mask][:, group].argmax(axis=1)
        base_acc = (base_pred == y_group[va_mask]).mean()
        acc_baseline.append(base_acc)
        
        # Train Ridge / Logistic Regression specialist
        clf = LogisticRegression(C=1.0, max_iter=300, random_state=42 + fold)
        clf.fit(X_full[tr_mask], y_group[tr_mask])
        
        spec_probs = clf.predict_proba(X_full[va_mask])
        oof_specialist_probs[name][va_mask] = spec_probs
        
        spec_pred = spec_probs.argmax(axis=1)
        spec_acc = (spec_pred == y_group[va_mask]).mean()
        acc_specialist.append(spec_acc)
        
    print(f"Specialist [{name:13s}]: Baseline Intra-Group Acc: {np.mean(acc_baseline):.4f} -> Specialist Acc: {np.mean(acc_specialist):.4f} (Delta: {np.mean(acc_specialist)-np.mean(acc_baseline):+.4f})")

# 5. Full Pipeline Simulation: Stage 1 (V68) + Stage 2 (Conditional Specialist Override)
print("\n=== STAGE 2 INTEGRATION ON V68 PIPELINE ===")

# Base V68 predictions
import sys
sys.path.append('.')
sys.path.append('lab')
from error_overlap_diagnostic import run_pipeline
pred_v68 = run_pipeline('V68', w_len=0.20, w_wlen=0.05, min_wlen_size=6, hungarian_anchor=True)
base_score = balanced_accuracy_score(y_true, pred_v68)
print(f"Base V68 Balanced Accuracy: {base_score:.4f}")

# Simulate Stage 2 override
for alpha in [0.20, 0.35, 0.50, 0.70, 1.00]:
    pred_stage2 = pred_v68.copy()
    n_modified = 0
    
    for name, group in SPECIALIST_GROUPS.items():
        # Condition: Top prediction is inside this specialist's group
        in_group_mask = np.isin(pred_v68, group)
        target_indices = np.where(in_group_mask)[0]
        
        for idx in target_indices:
            # Re-rank or adjust logits with specialist prob
            spec_p = oof_specialist_probs[name][idx]
            if spec_p.sum() > 0: # Valid OOF prediction
                chosen_local = spec_p.argmax()
                chosen_global = group[chosen_local]
                if alpha == 1.0:
                    pred_stage2[idx] = chosen_global
                    if chosen_global != pred_v68[idx]:
                        n_modified += 1
                else:
                    # Soft logit blending
                    # If specialist confidence is strong enough, override
                    if spec_p[chosen_local] >= 0.55 and chosen_global != pred_v68[idx]:
                        pred_stage2[idx] = chosen_global
                        n_modified += 1
                        
    score_stage2 = balanced_accuracy_score(y_true, pred_stage2)
    print(f"Stage 2 (alpha={alpha:4.2f}): Balanced Acc: {score_stage2:.4f} (Delta: {score_stage2 - base_score:+.4f}, Modified {n_modified} samples)")
