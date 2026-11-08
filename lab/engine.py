"""
NTL Digital Twin Lab: Core Simulation Engine
Implements 5-Fold GroupKFold validation, TF-IDF / Stylometrics OOF baseline,
and fast simulation of any post-processing pipeline.
"""
import os, re, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax

from .metrics import (
    LABELS, L2I, FAMILIES, LABEL_TO_FAMILY,
    balanced_accuracy_score, compute_confusion_analysis, bootstrap_confidence_interval
)
from .post_processors import (
    marginal_sinkhorn, evidence_clipping_gate,
    anchor_hungarian_assignment, soft_injective_dispersion, prompt_relative_length_bias
)


class SimulationEngine:
    def __init__(self, data_path='data/train.csv', cache_dir='lab/cache'):
        self.data_path = data_path
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.df = pd.read_csv(self.data_path)
        self.y_true = np.array([L2I[m] for m in self.df['model'].values], dtype=np.int32)
        self.groups = self.df['prompt_id'].values
        self.texts = self.df['text'].values
        self.n_samples = len(self.df)
        self.n_classes = len(LABELS)
        
        # Precompute text statistics
        self.word_counts = np.array([len(re.findall(r'\b\w+\b', str(t))) for t in self.texts], dtype=np.float32)
        self.char_counts = np.array([len(str(t)) for t in self.texts], dtype=np.float32)
        
        # Setup 5-Fold GroupKFold
        gkf = GroupKFold(n_splits=5)
        self.folds = list(gkf.split(self.df, self.y_true, groups=self.groups))
        
        self.base_probs = None
        self.base_evidence = None

    def build_or_load_base_oof(self, force_recompute=False):
        """
        Builds or loads the 5-fold Out-Of-Fold (OOF) base prediction matrix.
        Uses 4-View TF-IDF (word + char_wb) with calibrated LogisticRegression.
        """
        cache_file = os.path.join(self.cache_dir, 'base_oof.npz')
        if not force_recompute and os.path.exists(cache_file):
            print(f"[Lab] Loading cached OOF baseline from {cache_file}...")
            data = np.load(cache_file)
            self.base_probs = data['probs']
            self.base_evidence = data['evidence']
            return self.base_probs

        print("[Lab] Training 5-Fold GroupKFold TF-IDF Stylometrics Baseline...")
        start_t = time.time()
        
        oof_probs = np.zeros((self.n_samples, self.n_classes), dtype=np.float64)
        oof_evidence = np.zeros((self.n_samples, self.n_classes), dtype=np.float64)
        
        vec_word = TfidfVectorizer(ngram_range=(1, 2), max_features=35000, sublinear_tf=True)
        vec_char = TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', max_features=40000, sublinear_tf=True)
        
        from sklearn.linear_model import RidgeClassifier
        for fold, (train_idx, val_idx) in enumerate(self.folds):
            fold_t = time.time()
            X_tr_w = vec_word.fit_transform(self.texts[train_idx])
            X_va_w = vec_word.transform(self.texts[val_idx])
            
            X_tr_c = vec_char.fit_transform(self.texts[train_idx])
            X_va_c = vec_char.transform(self.texts[val_idx])
            
            import scipy.sparse as sp
            X_tr = sp.hstack([X_tr_w, X_tr_c]).tocsr()
            X_va = sp.hstack([X_va_w, X_va_c]).tocsr()
            
            clf = RidgeClassifier(alpha=1.5, random_state=42 + fold)
            clf.fit(X_tr, self.y_true[train_idx])
            
            val_logits = clf.decision_function(X_va)
            val_probs = softmax(val_logits, axis=1)
            
            oof_probs[val_idx] = val_probs
            oof_evidence[val_idx] = val_logits
            print(f"  Fold {fold+1}/5 completed in {time.time()-fold_t:.1f}s | Val Acc: {balanced_accuracy_score(self.y_true[val_idx], val_probs.argmax(axis=1)):.4f}", flush=True)

        np.savez_compressed(cache_file, probs=oof_probs, evidence=oof_evidence)
        self.base_probs = oof_probs
        self.base_evidence = oof_evidence
        print(f"[Lab] Baseline built in {time.time()-start_t:.1f}s. Raw OOF Balanced Acc: {balanced_accuracy_score(self.y_true, oof_probs.argmax(axis=1)):.4f}")
        return self.base_probs

    def simulate_pipeline(self,
                          sinkhorn_reg=0.60,
                          target_mix=0.35,
                          clip_val=1.2,
                          use_dispersion=True,
                          dispersion_damp=0.65,
                          use_length_bias=True,
                          use_anchor_hungarian=False,
                          anchor_margin=0.35,
                          anchor_prob=0.50):
        """
        Runs complete post-processing pipeline on OOF predictions.
        Executes in ~0.05 seconds!
        """
        if self.base_probs is None:
            self.build_or_load_base_oof()
            
        P = self.base_probs.copy()
        
        # Step 1: Prompt-Aware Relative Length Bias
        if use_length_bias:
            P = prompt_relative_length_bias(P, self.word_counts, self.groups)
            
        # Step 2: Evidence Clipping & Gating
        if clip_val > 0 and self.base_evidence is not None:
            raw_logits = np.log(np.maximum(P, 1e-12))
            gated_logits = evidence_clipping_gate(
                raw_logits, self.base_evidence,
                clip_val=clip_val, word_counts=self.word_counts
            )
            P = softmax(gated_logits, axis=1)
            
        # Step 3: Soft Injective Collision Dispersion
        if use_dispersion:
            P = soft_injective_dispersion(P, self.groups, damp=dispersion_damp)
            
        # Step 4: Marginal Sinkhorn Balancing
        if sinkhorn_reg > 0:
            P = marginal_sinkhorn(P, sinkhorn_reg=sinkhorn_reg, target_mix=target_mix)
            
        # Step 5: Decision / Hungarian
        if use_anchor_hungarian:
            y_pred = anchor_hungarian_assignment(P, self.groups,
                                                 anchor_margin=anchor_margin,
                                                 anchor_prob=anchor_prob)
        else:
            y_pred = P.argmax(axis=1)
            
        # Metrics
        acc = balanced_accuracy_score(self.y_true, y_pred)
        fold_accs = [balanced_accuracy_score(self.y_true[val], y_pred[val]) for _, val in self.folds]
        
        return {
            'balanced_accuracy': acc,
            'fold_accuracies': fold_accs,
            'fold_mean': float(np.mean(fold_accs)),
            'fold_std': float(np.std(fold_accs)),
            'y_pred': y_pred,
            'probs': P
        }

    def grid_search_sinkhorn(self, reg_values=None, mix_values=None):
        """
        Performs 2D Grid Search over Sinkhorn regularization and target mix.
        """
        if reg_values is None:
            reg_values = np.linspace(0.20, 0.80, 13)
        if mix_values is None:
            mix_values = np.linspace(0.15, 0.55, 9)
            
        best_score = -1.0
        best_params = {}
        results = []
        
        print(f"[Lab Grid Search] Scanning {len(reg_values)*len(mix_values)} Sinkhorn combinations...")
        t0 = time.time()
        for reg in reg_values:
            for mix in mix_values:
                res = self.simulate_pipeline(sinkhorn_reg=reg, target_mix=mix, clip_val=1.2)
                score = res['balanced_accuracy']
                results.append({'sinkhorn_reg': reg, 'target_mix': mix, 'score': score})
                if score > best_score:
                    best_score = score
                    best_params = {'sinkhorn_reg': reg, 'target_mix': mix}
                    
        print(f"[Lab Grid Search] Completed in {time.time()-t0:.2f}s!")
        print(f"-> Global Optimum: Sinkhorn = {best_params['sinkhorn_reg']:.2f}, Target Mix = {best_params['target_mix']:.2f} | Score = {best_score:.4f}")
        return best_params, best_score, pd.DataFrame(results)

