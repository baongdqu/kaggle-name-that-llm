import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def extract_stylometric_features(texts):
    """
    Trích xuất các đặc trưng văn phong thuần túy (Stylometry),
    hoàn toàn không phụ thuộc vào ngữ nghĩa/chủ đề (Topic-independent).
    """
    feats = []
    for t in texts:
        text = str(t)
        words = text.split()
        num_words = len(words) if len(words) > 0 else 1
        num_chars = len(text)
        
        # 1. Thống kê độ dài từ và câu
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 0]
        num_sentences = len(sentences) if len(sentences) > 0 else 1
        
        words_per_sentence = num_words / num_sentences
        chars_per_word = num_chars / num_words
        
        # 2. Thống kê dấu câu đặc trưng
        count_comma = text.count(',') / num_words
        count_period = text.count('.') / num_words
        count_semicolon = text.count(';') / num_words
        count_colon = text.count(':') / num_words
        count_hyphen = text.count('-') / num_words
        count_emdash = text.count('—') / num_words
        count_quote_single = text.count("'") / num_words
        count_quote_double = text.count('"') / num_words
        count_paren = (text.count('(') + text.count(')')) / num_words
        count_question = text.count('?') / num_words
        count_exclamation = text.count('!') / num_words
        
        # 3. Thống kê cấu trúc văn bản
        num_uppercase = sum(1 for c in text if c.isupper()) / (num_chars + 1)
        num_digits = sum(1 for c in text if c.isdigit()) / (num_chars + 1)
        
        # Tỷ lệ từ viết hoa đầu câu vs toàn từ viết hoa
        num_capitalized_words = sum(1 for w in words if w.istitle()) / num_words
        
        # Độ phong phú từ vựng (Type-Token Ratio - TTR)
        unique_words = len(set(w.lower() for w in words))
        ttr = unique_words / num_words
        
        # Tần suất các từ nối/liên từ đặc trưng (Discourse markers)
        lower_text = text.lower()
        has_furthermore = 1 if 'furthermore' in lower_text else 0
        has_moreover = 1 if 'moreover' in lower_text else 0
        has_however = 1 if 'however' in lower_text else 0
        has_therefore = 1 if 'therefore' in lower_text else 0
        has_in_summary = 1 if 'in summary' in lower_text or 'to summarize' in lower_text else 0
        has_specifically = 1 if 'specifically' in lower_text else 0
        has_essentially = 1 if 'essentially' in lower_text else 0
        has_ultimately = 1 if 'ultimately' in lower_text else 0
        has_crucially = 1 if 'crucially' in lower_text else 0
        
        row = [
            num_words,
            num_chars,
            num_sentences,
            words_per_sentence,
            chars_per_word,
            count_comma,
            count_period,
            count_semicolon,
            count_colon,
            count_hyphen,
            count_emdash,
            count_quote_single,
            count_quote_double,
            count_paren,
            count_question,
            count_exclamation,
            num_uppercase,
            num_digits,
            num_capitalized_words,
            ttr,
            has_furthermore,
            has_moreover,
            has_however,
            has_therefore,
            has_in_summary,
            has_specifically,
            has_essentially,
            has_ultimately,
            has_crucially
        ]
        feats.append(row)
        
    return np.array(feats, dtype=np.float32)

def run_experiment():
    print(">>> Đang nạp dữ liệu train...")
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    
    # Label encoding cho 18 models
    unique_models = sorted(train['model'].unique())
    model2id = {m: i for i, m in enumerate(unique_models)}
    id2model = {i: m for m, i in model2id.items()}
    y = train['model'].map(model2id).values
    groups = train['prompt_id'].values
    
    print(f"Tổng số mẫu: {len(train)}, Số class: {len(unique_models)}, Số prompts: {len(set(groups))}")
    
    # Trích xuất Stylometric features
    print(">>> Trích xuất Stylometric Features...")
    sty_feats = extract_stylometric_features(train['text'])
    
    # TF-IDF trên cấp độ ký tự (Char n-grams) và từ (Word n-grams)
    # Char n-grams (3-5) cực kỳ hiệu quả để bắt phong cách ghép từ/dấu câu của LLM mà ít bị ảnh hưởng bởi chủ đề
    print(">>> Trích xuất Char n-gram & Word n-gram TF-IDF...")
    char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=15000, sublinear_tf=True)
    word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=10000, sublinear_tf=True, stop_words='english')
    
    X_char = char_vec.fit_transform(train['text'])
    X_word = word_vec.fit_transform(train['text'])
    
    # Ghép toàn bộ feature matrix
    X_all = hstack([X_char, X_word, csr_matrix(sty_feats)]).tocsr()
    print(f"Shape ma trận đặc trưng kết hợp: {X_all.shape}")
    
    # Thiết lập 5-Fold Stratified Group K-Fold theo prompt_id (Chuẩn chống rò rỉ prompt)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros((len(train), len(unique_models)))
    
    print("\n>>> Bắt đầu chạy 5-Fold Stratified Group K-Fold Validation...")
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_all, y, groups=groups)):
        X_train, y_train = X_all[train_idx], y[train_idx]
        X_val, y_val = X_all[val_idx], y[val_idx]
        
        clf = LogisticRegression(C=2.5, max_iter=300, solver='lbfgs', random_state=42)
        clf.fit(X_train, y_train)
        
        probs = clf.predict_proba(X_val)
        oof_preds[val_idx] = probs
        
        preds = np.argmax(probs, axis=1)
        score = balanced_accuracy_score(y_val, preds)
        fold_scores.append(score)
        print(f"  Fold {fold+1}: Balanced Accuracy = {score:.4f}")
        
    overall_preds = np.argmax(oof_preds, axis=1)
    overall_score = balanced_accuracy_score(y, overall_preds)
    
    print(f"\n==========================================")
    print(f"🏆 OVERALL OOF BALANCED ACCURACY: {overall_score:.4f}")
    print(f"   Mean Fold Score: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
    print(f"==========================================")
    
    # Huấn luyện trên toàn bộ train và sinh file submission kiểm thử format
    print("\n>>> Fit toàn bộ tập train và xuất file submission.parquet...")
    clf_full = LogisticRegression(C=2.5, max_iter=300, solver='lbfgs', random_state=42)
    clf_full.fit(X_all, y)
    
    # Dự đoán test
    sty_test = extract_stylometric_features(test['text'])
    X_test_char = char_vec.transform(test['text'])
    X_test_word = word_vec.transform(test['text'])
    X_test = hstack([X_test_char, X_test_word, csr_matrix(sty_test)]).tocsr()
    
    test_probs = clf_full.predict_proba(X_test)
    test_pred_ids = np.argmax(test_probs, axis=1)
    test_pred_labels = [id2model[i] for i in test_pred_ids]
    
    sub = pd.DataFrame({
        'id': test['id'],
        'model': test_pred_labels
    })
    sub.to_parquet('submission.parquet', index=False)
    print(f"Đã xuất submission.parquet thành công ({len(sub)} dòng).")
    print("Dữ liệu nộp mẫu:")
    print(sub.head())

if __name__ == '__main__':
    run_experiment()
