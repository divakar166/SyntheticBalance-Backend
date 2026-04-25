from typing import Any, Dict

import pandas as pd


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    for col in normalized.columns:
        series = normalized[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            normalized[col] = series.map(
                lambda value: value.strip() if isinstance(value, str) else value
            ).replace("", pd.NA)

    return normalized


class SchemaDetector:
    """Auto-detect schema from CSV (numeric vs categorical, cardinality, types)"""
    
    NUMERIC_THRESHOLD = 0.9  # If >90% values are numeric, treat as numeric
    CATEGORICAL_THRESHOLD = 20  # If <20 unique values, treat as categorical
    
    @staticmethod
    def detect(df: pd.DataFrame, target_col: str = None) -> Dict[str, Dict]:
        """
        Args:
            df: pandas DataFrame
            target_col: name of target/label column (optional)
            
        Returns:
            schema = {
                'features': {
                    'age': {'type': 'numeric', 'min': 18, 'max': 80},
                    'country': {'type': 'categorical', 'cardinality': 45, 'top_3': ['US', 'UK', 'DE']}
                },
                'target': {'type': 'categorical', 'cardinality': 2, 'class_dist': {0: 9500, 1: 500}}
            }
        """
        df = normalize_dataframe(df)
        schema = {'features': {}, 'target': None, 'skipped_features': []}

        cols = df.columns.tolist()
        if target_col and target_col in cols:
            cols.remove(target_col)

        for col in cols:
            series = df[col]
            profile = SchemaDetector._profile_feature(series)
            if profile is None:
                schema['skipped_features'].append(col)
                continue
            schema['features'][col] = profile

        if target_col and target_col in df.columns:
            target_vals = df[target_col]
            non_null = target_vals.dropna()
            if not non_null.empty:
                class_distribution = {
                    str(SchemaDetector._serialize_scalar(label)): int(count)
                    for label, count in non_null.value_counts().items()
                }
                schema['target'] = {
                    'name': target_col,
                    'type': 'categorical',
                    'cardinality': int(non_null.nunique()),
                    'missing_count': int(target_vals.isna().sum()),
                    'missing_pct': float(target_vals.isna().mean() * 100),
                    'class_distribution': class_distribution
                }

        return schema

    @staticmethod
    def _profile_feature(series: pd.Series) -> Dict[str, Any] | None:
        series = normalize_dataframe(pd.DataFrame({"value": series}))["value"]
        total_count = len(series)
        missing_count = int(series.isna().sum())
        missing_pct = float((missing_count / total_count) * 100) if total_count else 0.0
        non_null = series.dropna()

        if non_null.empty:
            return None

        coerced_numeric = pd.to_numeric(non_null, errors='coerce')
        numeric_ratio = float(coerced_numeric.notna().mean())
        unique_count = int(non_null.nunique())
        is_constant = unique_count <= 1

        base_profile: Dict[str, Any] = {
            'missing_count': missing_count,
            'missing_pct': missing_pct,
            'unique_values': unique_count,
            'is_constant': is_constant,
            'min': None,
            'max': None,
            'mean': None,
            'std': None,
            'median': None,
            'q1': None,
            'q3': None,
            'iqr': None,
            'skewness': None,
            'kurtosis': None,
            'cardinality': None,
            'top_values': [],
            'top_value_stats': [],
            'example_values': [],
            'most_common_freq': None,
        }

        if numeric_ratio >= SchemaDetector.NUMERIC_THRESHOLD:
            q1 = coerced_numeric.quantile(0.25)
            q3 = coerced_numeric.quantile(0.75)
            return {
                **base_profile,
                'type': 'numeric',
                'min': SchemaDetector._safe_float(coerced_numeric.min()),
                'max': SchemaDetector._safe_float(coerced_numeric.max()),
                'mean': SchemaDetector._safe_float(coerced_numeric.mean()),
                'std': SchemaDetector._safe_float(coerced_numeric.std()),
                'median': SchemaDetector._safe_float(coerced_numeric.median()),
                'q1': SchemaDetector._safe_float(q1),
                'q3': SchemaDetector._safe_float(q3),
                'iqr': SchemaDetector._safe_float(q3 - q1),
                'skewness': SchemaDetector._safe_float(coerced_numeric.skew()),
                'kurtosis': SchemaDetector._safe_float(coerced_numeric.kurt()),
            }

        value_counts = non_null.value_counts().head(5)
        top_value_stats = []
        for value, count in value_counts.items():
            top_value_stats.append(
                {
                    'value': str(SchemaDetector._serialize_scalar(value)),
                    'count': int(count),
                    'pct': float((count / len(non_null)) * 100),
                }
            )

        return {
            **base_profile,
            'type': 'categorical',
            'cardinality': unique_count,
            'top_values': [item['value'] for item in top_value_stats],
            'top_value_stats': top_value_stats,
            'example_values': [str(SchemaDetector._serialize_scalar(value)) for value in non_null.head(3).tolist()],
            'most_common_freq': int(value_counts.iloc[0]) if not value_counts.empty else 0,
        }

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _serialize_scalar(value: Any) -> Any:
        if hasattr(value, 'item'):
            return value.item()
        return value


class Preprocessor:
    """Normalize numerics, encode categoricals, handle missing values"""
    
    def __init__(self, schema: Dict):
        self.schema = schema
        self.numeric_scaler = {}  # min/max per numeric feature
        self.categorical_encoder = {}  # category → index mapping
        self.fitted = False
        
    def fit(self, df: pd.DataFrame):
        """Learn statistics from training data"""
        for col, col_schema in self.schema['features'].items():
            if col_schema['type'] == 'numeric':
                # Min-max normalization
                self.numeric_scaler[col] = {
                    'min': df[col].min(),
                    'max': df[col].max()
                }
            else:
                # Category encoding
                unique_cats = df[col].dropna().unique()
                self.categorical_encoder[col] = {
                    cat: idx for idx, cat in enumerate(unique_cats)
                }
        self.fitted = True
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization/encoding"""
        assert self.fitted, "Call fit() first"
        
        df_proc = df.copy()
        
        for col in df_proc.columns:
            if col not in self.schema['features']:
                continue
                
            col_schema = self.schema['features'][col]
            
            if col_schema['type'] == 'numeric':
                # Normalize to [0, 1]
                min_val = self.numeric_scaler[col]['min']
                max_val = self.numeric_scaler[col]['max']
                df_proc[col] = (df_proc[col] - min_val) / (max_val - min_val + 1e-8)
                df_proc[col] = df_proc[col].clip(0, 1)  # Ensure [0, 1]
                
            else:
                # Encode categories
                mapping = self.categorical_encoder[col]
                df_proc[col] = df_proc[col].map(mapping)
        
        return df_proc
    
    def inverse_transform(self, df_proc: pd.DataFrame) -> pd.DataFrame:
        """Denormalize back to original scale"""
        df = df_proc.copy()
        
        for col in df.columns:
            if col not in self.schema['features']:
                continue
                
            col_schema = self.schema['features'][col]
            
            if col_schema['type'] == 'numeric':
                min_val = self.numeric_scaler[col]['min']
                max_val = self.numeric_scaler[col]['max']
                df[col] = df[col] * (max_val - min_val) + min_val
                
            else:
                # Reverse encoding
                reverse_map = {
                    idx: cat for cat, idx in 
                    self.categorical_encoder[col].items()
                }
                df[col] = df[col].map(reverse_map)
        
        return df
