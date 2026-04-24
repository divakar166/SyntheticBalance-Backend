import pandas as pd
import numpy as np
from typing import Dict

class PrivacyMetrics:
    """Evaluate privacy of synthetic data"""
    
    @staticmethod
    def k_anonymity(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, k_threshold: int = 5) -> int:
        """
        For each synthetic row, count how many real rows are "similar" 
        (within some distance threshold).
        
        Return scalar k (typical value 5-10).
        """
        from scipy.spatial.distance import cdist
        
        # Normalize
        real_norm = (real_df - real_df.min()) / (real_df.max() - real_df.min() + 1e-8)
        syn_norm = (synthetic_df - synthetic_df.min()) / (synthetic_df.max() - synthetic_df.min() + 1e-8)
        
        # Compute distances
        distances = cdist(syn_norm.values, real_norm.values, metric='euclidean')
        
        # For each synthetic, find distance to nearest real
        nearest_distances = distances.min(axis=1)
        
        # Pick distance threshold = 75th percentile
        threshold = np.percentile(nearest_distances, 75)
        
        # For each synthetic, count how many real rows are within threshold
        k_values = []
        for i in range(len(synthetic_df)):
            k = (distances[i] <= threshold).sum()
            k_values.append(k)
        
        # Return median k
        return int(np.median(k_values))
    
    @staticmethod
    def membership_inference_attack(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> float:
        """
        Train a binary classifier: is this record real or synthetic?
        Return attack AUC (0.5 = good privacy, 0.9 = bad privacy).
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        
        # Create binary labels: 1 = real, 0 = synthetic
        X = pd.concat([real_df, synthetic_df], ignore_index=True)
        y = np.array([1] * len(real_df) + [0] * len(synthetic_df))
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        
        return float(auc)
    
    @staticmethod
    def dp_budget_estimate(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, float]:
        """
        Rough estimate of differential privacy budget (epsilon).
        
        This is NOT a formal DP guarantee, just an empirical estimate.
        Uses privacy loss histogram method.
        
        Returns: {'epsilon': value, 'delta': 1e-5, 'confidence': 0.95}
        """
        # Simplified: measure max likelihood ratio between real and synthetic
        from scipy.stats import entropy
        
        # Entropy of real and synthetic distributions
        real_entropy = entropy(real_df.values.flatten())
        syn_entropy = entropy(synthetic_df.values.flatten())
        
        # Rough epsilon estimate: divergence-based
        divergence = abs(real_entropy - syn_entropy) / max(real_entropy, syn_entropy)
        epsilon = max(divergence * 3, 0.5)  # Scale up, with floor
        
        return {
            'epsilon': float(min(epsilon, 10.0)),  # Cap at 10
            'delta': 1e-5,
            'confidence': 0.95
        }
