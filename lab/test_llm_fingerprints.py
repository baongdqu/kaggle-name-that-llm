import sys, os, re
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import spearmanr

train = pd.read_csv('data/train.csv')
labels = sorted(train.model.unique())
l2i = {l: i for i, l in enumerate(labels)}
y_true = np.array([l2i[m] for m in train.model.values])
prompts = train.prompt_id.values

print("=== LLM FINGERPRINTING & RELATIVE RANK LAB ===")
print(f"Total training samples: {len(train)}")
print(f"Total unique prompts: {len(np.unique(prompts))}")

# 1. Feature Engineering: Domain LLM Stylistic Signatures
def extract_llm_fingerprints(df):
    features = []
    for text in df.text.values:
        st = str(text)
        words = st.split()
        nw = max(1, len(words))
        chars = len(st)
        lines = st.split('\n')
        nl = max(1, len(lines))
        
        # --- A. Markdown & Formatting Signatures (GPT vs Claude vs Gemini) ---
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', st)) / nw
        italic_count = len(re.findall(r'(?<!\*)\*[^*]+\*(?!\*)', st)) / nw
        bullet_count = len(re.findall(r'^\s*[-*•]\s+', st, re.MULTILINE)) / nl
        numbered_list = len(re.findall(r'^\s*\d+[\.)]\s+', st, re.MULTILINE)) / nl
        header_count = len(re.findall(r'^\s*#{1,6}\s+', st, re.MULTILINE)) / nl
        code_block_count = st.count('```')
        inline_code_count = len(re.findall(r'`[^`]+`', st))
        
        # --- B. Specific Punctuation & Typography Habits ---
        emdash_count = (st.count('—') + st.count('–')) / nw
        ellipsis_count = (st.count('...') + st.count('…')) / nw
        colon_count = st.count(':') / nw
        semicolon_count = st.count(';') / nw
        exclamation_count = st.count('!') / nw
        question_count = st.count('?') / nw
        parentheses_count = st.count('(') / nw
        quote_double = (st.count('"') + st.count('“') + st.count('”')) / nw
        quote_single = (st.count("'") + st.count('‘') + st.count('’')) / nw
        
        # --- C. Lexical Diversity & Word Length Profile ---
        unique_words = len(set(w.lower() for w in words))
        ttr = unique_words / nw # Type-Token Ratio
        avg_word_len = chars / nw
        long_words = sum(1 for w in words if len(w) >= 8) / nw
        short_words = sum(1 for w in words if len(w) <= 3) / nw
        
        # --- D. Sentence-Level Variance & Rhythm ---
        sents = [s.strip() for s in re.split(r'[.!?]+', st) if s.strip()]
        ns = max(1, len(sents))
        sent_lens = np.array([len(s.split()) for s in sents], dtype=np.float32)
        mean_sent_len = float(np.mean(sent_lens))
        std_sent_len = float(np.std(sent_lens)) if len(sent_lens) > 1 else 0.0
        max_sent_len = float(np.max(sent_lens))
        
        # --- E. Specific Discourse Markers & Boilerplate Tells ---
        first_person = len(re.findall(r'\b(i|me|my|mine|we|us|our)\b', st, re.I)) / nw
        second_person = len(re.findall(r'\b(you|your|yours)\b', st, re.I)) / nw
        contrastive_markers = len(re.findall(r'\b(however|although|nevertheless|conversely|whereas|on the other hand)\b', st, re.I)) / nw
        causal_markers = len(re.findall(r'\b(therefore|thus|consequently|hence|accordingly|as a result)\b', st, re.I)) / nw
        hedging_markers = len(re.findall(r'\b(perhaps|possibly|probably|likely|arguably|seem|seems|suggests)\b', st, re.I)) / nw
        
        features.append([
            bold_count, italic_count, bullet_count, numbered_list, header_count,
            code_block_count, inline_code_count, emdash_count, ellipsis_count,
            colon_count, semicolon_count, exclamation_count, question_count,
            parentheses_count, quote_double, quote_single, ttr, avg_word_len,
            long_words, short_words, mean_sent_len, std_sent_len, max_sent_len,
            first_person, second_person, contrastive_markers, causal_markers, hedging_markers
        ])
    return np.array(features, dtype=np.float32)

print("\nExtracting 28-dimensional LLM Fingerprint features...")
F_raw = extract_llm_fingerprints(train)
print(f"Extracted shape: {F_raw.shape}")

# 2. Per-Prompt Relative Ranking & Z-Score Normalization
print("Computing Intra-Prompt Relative Features...")
F_rel = np.zeros_like(F_raw)
for p in np.unique(prompts):
    idx = np.where(prompts == p)[0]
    if len(idx) > 1:
        sub = F_raw[idx]
        mean = sub.mean(axis=0)
        std = sub.std(axis=0) + 1e-6
        F_rel[idx] = (sub - mean) / std
    else:
        F_rel[idx] = 0.0

# 3. Analyze Feature Power Across Model Families
FAMILY_MAP = {
    'claude': [0, 1, 2, 3, 4], # fable, opus-4-6, opus-4-8, opus-5, sonnet-5
    'gemini': [5, 6, 7, 8],    # 3.1-pro, 3.6-flash, 3.7-flash, 3.8-flash
    'glm': [9, 10],           # 5.3, 5.3-flash
    'gpt': [11, 12, 13, 14, 15], # 5.5, luna, sol, terra, astra
    'kimi': [16],
    'qwen': [17]
}

FEAT_NAMES = [
    'bold_rate', 'italic_rate', 'bullet_rate', 'numbered_rate', 'header_rate',
    'code_blocks', 'inline_code', 'emdash_rate', 'ellipsis_rate',
    'colon_rate', 'semicolon_rate', 'exclamation_rate', 'question_rate',
    'parentheses_rate', 'quote_double', 'quote_single', 'ttr', 'avg_word_len',
    'long_words', 'short_words', 'mean_sent_len', 'std_sent_len', 'max_sent_len',
    'first_person', 'second_person', 'contrastive', 'causal', 'hedging'
]

print("\n=== TOP DISTINCTIVE FINGERPRINT SIGNATURES BY FAMILY ===")
for fam, c_idxs in FAMILY_MAP.items():
    fam_mask = np.isin(y_true, c_idxs)
    diffs = F_raw[fam_mask].mean(axis=0) - F_raw[~fam_mask].mean(axis=0)
    top_pos = np.argsort(-diffs)[:3]
    top_neg = np.argsort(diffs)[:2]
    print(f"\nFamily [{fam.upper():6s}] (N={fam_mask.sum()}):")
    print(f"  (+) Top Characteristics: {', '.join([f'{FEAT_NAMES[i]} (+{diffs[i]:.4f})' for i in top_pos])}")
    print(f"  (-) Lowest Presence    : {', '.join([f'{FEAT_NAMES[i]} ({diffs[i]:.4f})' for i in top_neg])}")

