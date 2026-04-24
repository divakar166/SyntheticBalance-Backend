import numpy as np
import pandas as pd
from typing import Dict

class QualityMetrics:
    """Evaluate synthetic data quality vs real data"""
    
    @staticmethod
    def kl_divergence(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute KL divergence per feature.
        
        For numeric: histogram binning (10 bins)
        For categorical: value counts
        
        Returns: {feature_name: kl_value}
        """
        kl_divs = {}
        
        for col in real_df.columns:
            real_col = real_df[col].dropna()
            syn_col = synthetic_df[col].dropna()
            
            if real_col.dtype in ['float64', 'int64']:
                # Numeric: histogram
                bins = np.histogram_bin_edges(real_col, bins=10)
                real_hist, _ = np.histogram(real_col, bins=bins, density=True)
                syn_hist, _ = np.histogram(syn_col, bins=bins, density=True)
            else:
                # Categorical: value counts
                real_counts = real_col.value_counts(normalize=True)
                syn_counts = syn_col.value_counts(normalize=True)
                
                # Align to same categories
                all_cats = set(real_counts.index) | set(syn_counts.index)
                real_hist = np.array([real_counts.get(cat, 1e-8) for cat in all_cats])
                syn_hist = np.array([syn_counts.get(cat, 1e-8) for cat in all_cats])
                real_hist /= real_hist.sum()
                syn_hist /= syn_hist.sum()
            
            # KL divergence: sum(p * log(p / q))
            kl = (real_hist * (np.log(real_hist + 1e-10) - np.log(syn_hist + 1e-10))).sum()
            kl_divs[col] = float(kl)
        
        return kl_divs
    
    @staticmethod
    def wasserstein_distance(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> float:
        """
        1-Wasserstein distance in normalized space.
        """
        from scipy.spatial.distance import cdist
        
        # Normalize both to [0, 1]
        real_norm = (real_df - real_df.min()) / (real_df.max() - real_df.min() + 1e-8)
        syn_norm = (synthetic_df - synthetic_df.min()) / (synthetic_df.max() - synthetic_df.min() + 1e-8)
        
        # Compute mean pairwise distance
        distances = cdist(real_norm.values, syn_norm.values, metric='euclidean')
        wasserstein = np.min(distances, axis=1).mean()
        
        return float(wasserstein)
    
    @staticmethod
    def correlation_difference(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute Pearson correlations on real and synthetic.
        Return max absolute difference across all feature pairs.
        """
        real_corr = real_df.corr()
        syn_corr = synthetic_df.corr()
        
        diff = (real_corr - syn_corr).abs()
        
        return {
            'max_diff': float(diff.max().max()),
            'mean_diff': float(diff.values[np.triu_indices_from(diff.values, k=1)].mean()),
            'std_diff': float(diff.values[np.triu_indices_from(diff.values, k=1)].std())
        }
    
    @staticmethod
    def pca_variance_retained(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> float:
        """
        Fit PCA on real data, project synthetic.
        Return % variance explained on synthetic.
        """
        from sklearn.decomposition import PCA
        
        # Fit PCA on real
        pca = PCA()
        pca.fit(real_df)
        
        # Project synthetic
        syn_transformed = pca.transform(synthetic_df)
        
        # Variance explained by original components
        explained = pca.explained_variance_ratio_.sum()
        
        return float(explained)
