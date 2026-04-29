import numpy as np
import pandas as pd
from typing import Dict

class QualityMetrics:
    """Evaluate synthetic data quality vs real data"""

    @staticmethod
    def _get_numeric(df):
        return df.select_dtypes(include=["number"])

    @staticmethod
    def _common_numeric(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        real_cols = []
        for col in real_df.columns.intersection(synthetic_df.columns):
            real_col = pd.to_numeric(real_df[col], errors="coerce")
            syn_col = pd.to_numeric(synthetic_df[col], errors="coerce")
            real_ratio = real_col.notna().mean() if len(real_col) else 0.0
            syn_ratio = syn_col.notna().mean() if len(syn_col) else 0.0
            dtype_numeric = (
                pd.api.types.is_numeric_dtype(real_df[col])
                or pd.api.types.is_numeric_dtype(synthetic_df[col])
            )
            if dtype_numeric or (real_ratio >= 0.8 and syn_ratio >= 0.8):
                if real_col.notna().any() and syn_col.notna().any():
                    real_cols.append(col)

        return (
            real_df[real_cols].apply(pd.to_numeric, errors="coerce"),
            synthetic_df[real_cols].apply(pd.to_numeric, errors="coerce"),
        )
    
    @staticmethod
    def kl_divergence(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute KL divergence per feature.
        
        For numeric: histogram binning (10 bins)
        For categorical: value counts
        
        Returns: {feature_name: kl_value}
        """
        kl_divs = {}
        
        for col in real_df.columns.intersection(synthetic_df.columns):
            real_col = real_df[col].dropna()
            syn_col = synthetic_df[col].dropna()
            if real_col.empty or syn_col.empty:
                kl_divs[col] = 0.0
                continue
            
            real_num = pd.to_numeric(real_col, errors="coerce")
            syn_num = pd.to_numeric(syn_col, errors="coerce")
            is_numeric = (
                pd.api.types.is_numeric_dtype(real_col)
                or pd.api.types.is_numeric_dtype(syn_col)
            )
            if is_numeric and real_num.notna().any() and syn_num.notna().any():
                # Numeric: histogram
                real_values = real_num.dropna().to_numpy(dtype=float)
                syn_values = syn_num.dropna().to_numpy(dtype=float)
                combined = np.concatenate([real_values, syn_values])
                if np.nanmin(combined) == np.nanmax(combined):
                    kl_divs[col] = 0.0
                    continue

                bins = np.histogram_bin_edges(combined, bins=10)
                real_hist, _ = np.histogram(real_values, bins=bins, density=False)
                syn_hist, _ = np.histogram(syn_values, bins=bins, density=False)

                real_hist = real_hist + 1e-10
                syn_hist = syn_hist + 1e-10
                real_hist = real_hist / real_hist.sum()
                syn_hist = syn_hist / syn_hist.sum()
            else:
                # Categorical: value counts
                real_counts = real_col.astype(str).value_counts(normalize=True)
                syn_counts = syn_col.astype(str).value_counts(normalize=True)
                
                # Align to same categories
                all_cats = set(real_counts.index) | set(syn_counts.index)
                real_hist = np.array([real_counts.get(cat, 1e-8) for cat in all_cats])
                syn_hist = np.array([syn_counts.get(cat, 1e-8) for cat in all_cats])
                real_hist /= real_hist.sum()
                syn_hist /= syn_hist.sum()
            
            # KL divergence: sum(p * log(p / q))
            kl = (real_hist * (np.log(real_hist + 1e-10) - np.log(syn_hist + 1e-10))).sum()

            kl_divs[col] = float(kl) if np.isfinite(kl) else 0.0
        
        return kl_divs
    
    @staticmethod
    def wasserstein_distance(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> float:
        """
        Mean per-column 1-Wasserstein distance in normalized numeric space.
        """
        from scipy.stats import wasserstein_distance

        real_num, syn_num = QualityMetrics._common_numeric(real_df, synthetic_df)

        if real_num.empty or syn_num.empty:
            return 0.0

        distances = []
        for col in real_num.columns:
            real_values = real_num[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            syn_values = syn_num[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            if len(real_values) == 0 or len(syn_values) == 0:
                continue

            real_min = float(np.min(real_values))
            real_max = float(np.max(real_values))
            scale = real_max - real_min
            if scale <= 1e-12:
                scale = max(abs(real_min), abs(float(np.max(syn_values))), abs(float(np.min(syn_values))), 1.0)

            real_norm = (real_values - real_min) / scale
            syn_norm = (syn_values - real_min) / scale
            distance = wasserstein_distance(real_norm, syn_norm)
            if np.isfinite(distance):
                distances.append(float(distance))

        return float(np.mean(distances)) if distances else 0.0
    
    @staticmethod
    def correlation_difference(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute Pearson correlations on real and synthetic.
        Return max absolute difference across all feature pairs.
        """
        real_num, syn_num = QualityMetrics._common_numeric(real_df, synthetic_df)
        if len(real_num.columns) < 2 or real_num.empty or syn_num.empty:
            return {'max_diff': 0.0, 'mean_diff': 0.0, 'std_diff': 0.0}

        fill_values = real_num.median(numeric_only=True).fillna(0.0)
        real_num = real_num.replace([np.inf, -np.inf], np.nan).fillna(fill_values)
        syn_num = syn_num.replace([np.inf, -np.inf], np.nan).fillna(fill_values)

        real_corr = real_num.corr().fillna(0.0)
        syn_corr = syn_num.corr().fillna(0.0)
        
        diff = (real_corr - syn_corr).abs()
        upper = diff.values[np.triu_indices_from(diff.values, k=1)]
        
        return {
            'max_diff': float(diff.max().max()),
            'mean_diff': float(upper.mean()) if len(upper) else 0.0,
            'std_diff': float(upper.std()) if len(upper) else 0.0
        }
    
    @staticmethod
    def pca_variance_retained(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> float:
        """
        Fit PCA on real data, project synthetic.
        Return % variance explained on synthetic.
        """
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        real_num, syn_num = QualityMetrics._common_numeric(real_df, synthetic_df)

        if real_num.empty or syn_num.empty:
            return 0.0

        fill_values = real_num.median(numeric_only=True).fillna(0.0)
        real_num = real_num.replace([np.inf, -np.inf], np.nan).fillna(fill_values)
        syn_num = syn_num.replace([np.inf, -np.inf], np.nan).fillna(fill_values)
        n_components = min(real_num.shape[0], real_num.shape[1])
        if n_components < 1:
            return 0.0

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(imputer.fit_transform(real_num))
        syn_scaled = scaler.transform(imputer.transform(syn_num))

        pca_full = PCA(n_components=n_components)
        pca_full.fit(real_scaled)
        cumulative = np.cumsum(pca_full.explained_variance_ratio_)
        retained_components = int(np.searchsorted(cumulative, 0.95) + 1)
        retained_components = max(1, min(retained_components, n_components))

        pca = PCA(n_components=retained_components)
        pca.fit(real_scaled)
        syn_projected = pca.transform(syn_scaled)
        
        total_variance = float(np.var(syn_scaled, axis=0, ddof=0).sum())
        if total_variance <= 1e-12:
            return 1.0

        retained_variance = float(np.var(syn_projected, axis=0, ddof=0).sum())
        variance = retained_variance / total_variance
        return float(min(max(variance, 0.0), 1.0)) if np.isfinite(variance) else 0.0
