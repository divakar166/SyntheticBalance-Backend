import pandas as pd
import numpy as np
from typing import Dict

class PrivacyMetrics:
    """Evaluate privacy of synthetic data"""

    @staticmethod
    def _get_numeric(df):
        return df.select_dtypes(include=["number"])

    @staticmethod
    def _common_numeric(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        cols = []
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
                    cols.append(col)

        return (
            real_df[cols].apply(pd.to_numeric, errors="coerce"),
            synthetic_df[cols].apply(pd.to_numeric, errors="coerce"),
        )

    @staticmethod
    def _normalized_numeric_matrices(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        real_num, syn_num = PrivacyMetrics._common_numeric(real_df, synthetic_df)
        if real_num.empty or syn_num.empty:
            return np.empty((0, 0)), np.empty((0, 0))

        fill_values = real_num.replace([np.inf, -np.inf], np.nan).median(numeric_only=True).fillna(0.0)
        real_num = real_num.replace([np.inf, -np.inf], np.nan).fillna(fill_values)
        syn_num = syn_num.replace([np.inf, -np.inf], np.nan).fillna(fill_values)

        mins = real_num.min()
        ranges = (real_num.max() - mins).replace(0, 1.0).fillna(1.0)
        real_norm = ((real_num - mins) / ranges).clip(-5, 5)
        syn_norm = ((syn_num - mins) / ranges).clip(-5, 5)
        return real_norm.to_numpy(dtype=float), syn_norm.to_numpy(dtype=float)
    
    @staticmethod
    def k_anonymity(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, k_threshold: int = 5) -> int:
        """
        For each synthetic row, count how many real rows are "similar" 
        (within some distance threshold).
        
        Return scalar k (typical value 5-10).
        """
        from scipy.spatial.distance import cdist

        real_matrix, syn_matrix = PrivacyMetrics._normalized_numeric_matrices(real_df, synthetic_df)
        if real_matrix.size == 0 or syn_matrix.size == 0:
            common_cats = [
                col for col in real_df.columns.intersection(synthetic_df.columns)
                if (
                    not pd.api.types.is_numeric_dtype(real_df[col])
                    and not pd.api.types.is_numeric_dtype(synthetic_df[col])
                )
            ]
            if not common_cats:
                return 0

            real_keys = (
                real_df[common_cats]
                .fillna("__missing__")
                .astype(str)
                .agg("\x1f".join, axis=1)
                .value_counts()
            )
            syn_keys = (
                synthetic_df[common_cats]
                .fillna("__missing__")
                .astype(str)
                .agg("\x1f".join, axis=1)
            )
            counts = syn_keys.map(real_keys).fillna(0).to_numpy(dtype=float)
            return int(np.median(counts)) if len(counts) else 0
        
        # Compute distances
        distances = cdist(syn_matrix, real_matrix, metric='euclidean')
        distances = distances[np.isfinite(distances).all(axis=1)]
        if distances.size == 0:
            return 0
        
        # For each synthetic, find distance to nearest real
        nearest_distances = distances.min(axis=1)
        nearest_distances = nearest_distances[np.isfinite(nearest_distances)]
        if len(nearest_distances) == 0:
            return 0
        
        # Pick distance threshold = 75th percentile
        threshold = np.percentile(nearest_distances, 75)
        
        # For each synthetic, count how many real rows are within threshold
        k_values = []
        for i in range(len(distances)):
            k = (distances[i] <= threshold).sum()
            k_values.append(k)
        
        # Return median k
        return int(np.median(k_values)) if k_values else 0
    
    @staticmethod
    def membership_inference_attack(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> float:
        """
        Train a binary classifier: is this record real or synthetic?
        Return attack AUC (0.5 = good privacy, 0.9 = bad privacy).
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        common_cols = list(real_df.columns.intersection(synthetic_df.columns))
        if not common_cols or len(real_df) < 2 or len(synthetic_df) < 2:
            return 0.5

        # Create binary labels: 1 = real, 0 = synthetic
        X = pd.concat([real_df[common_cols], synthetic_df[common_cols]], ignore_index=True)
        X = pd.get_dummies(X.replace([np.inf, -np.inf], np.nan), dummy_na=True)
        y = np.array([1] * len(real_df) + [0] * len(synthetic_df))
        if X.empty or len(np.unique(y)) < 2:
            return 0.5
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            return 0.5
        
        # Train classifier
        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("model", RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
                min_samples_leaf=2,
            )),
        ])
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        
        return float(auc) if np.isfinite(auc) else 0.5
    
    @staticmethod
    def dp_budget_estimate(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, float]:
        """
        Rough estimate of differential privacy budget (epsilon).
        
        This is NOT a formal DP guarantee, just an empirical estimate.
        Uses privacy loss histogram method.
        
        Returns: {'epsilon': value, 'delta': 1e-5, 'confidence': 0.95}
        """
        divergences = []

        for col in real_df.columns.intersection(synthetic_df.columns):
            real_col = real_df[col].dropna()
            syn_col = synthetic_df[col].dropna()
            if real_col.empty or syn_col.empty:
                continue

            real_num = pd.to_numeric(real_col, errors="coerce")
            syn_num = pd.to_numeric(syn_col, errors="coerce")
            is_numeric = (
                pd.api.types.is_numeric_dtype(real_df[col])
                or pd.api.types.is_numeric_dtype(synthetic_df[col])
            )

            if is_numeric and real_num.notna().any() and syn_num.notna().any():
                real_values = real_num.dropna().to_numpy(dtype=float)
                syn_values = syn_num.dropna().to_numpy(dtype=float)
                combined = np.concatenate([real_values, syn_values])
                if np.nanmin(combined) == np.nanmax(combined):
                    divergences.append(0.0)
                    continue
                bins = np.histogram_bin_edges(combined, bins=20)
                real_hist, _ = np.histogram(real_values, bins=bins)
                syn_hist, _ = np.histogram(syn_values, bins=bins)
            else:
                real_counts = real_col.astype(str).value_counts()
                syn_counts = syn_col.astype(str).value_counts()
                categories = sorted(set(real_counts.index) | set(syn_counts.index))
                real_hist = np.array([real_counts.get(cat, 0) for cat in categories], dtype=float)
                syn_hist = np.array([syn_counts.get(cat, 0) for cat in categories], dtype=float)

            real_prob = (real_hist.astype(float) + 1e-9)
            syn_prob = (syn_hist.astype(float) + 1e-9)
            real_prob = real_prob / real_prob.sum()
            syn_prob = syn_prob / syn_prob.sum()
            tv_distance = 0.5 * np.abs(real_prob - syn_prob).sum()
            if np.isfinite(tv_distance):
                divergences.append(float(tv_distance))

        mean_tv = float(np.mean(divergences)) if divergences else 0.0
        mean_tv = min(max(mean_tv, 0.0), 1.0 - 1e-9)
        epsilon = np.log((1.0 + mean_tv) / (1.0 - mean_tv))

        return {
            'epsilon': float(min(max(epsilon, 0.0), 10.0)),
            'delta': 1e-5,
            'confidence': 0.95
        }
