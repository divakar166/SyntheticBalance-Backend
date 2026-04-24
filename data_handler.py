import pandas as pd
from typing import Dict, List, Optional

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
        schema = {'features': {}, 'target': None}
        
        cols = df.columns.tolist()
        if target_col:
            cols.remove(target_col)
        
        for col in cols:
            try:
                # Try numeric conversion
                numeric_vals = pd.to_numeric(df[col], errors='coerce')
                numeric_ratio = numeric_vals.notna().sum() / len(df)
                
                if numeric_ratio >= SchemaDetector.NUMERIC_THRESHOLD:
                    schema['features'][col] = {
                        'type': 'numeric',
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'missing_pct': float(df[col].isna().sum() / len(df) * 100)
                    }
                else:
                    raise ValueError("Not numeric")
            except (ValueError, TypeError):
                # Treat as categorical
                cardinality = df[col].nunique()
                top_values = df[col].value_counts().head(5)
                
                schema['features'][col] = {
                    'type': 'categorical',
                    'cardinality': int(cardinality),
                    'top_values': top_values.index.tolist(),
                    'missing_pct': float(df[col].isna().sum() / len(df) * 100)
                }
        
        if target_col:
            target_vals = df[target_col]
            cardinality = target_vals.nunique()

            if cardinality <= SchemaDetector.CATEGORICAL_THRESHOLD:
                schema['target'] = {
                    'name': target_col,
                    'type': 'categorical',
                    'cardinality': int(cardinality),
                    'class_distribution': target_vals.value_counts().to_dict()
                }
            else:
                # Try numeric fallback if possible
                numeric_vals = pd.to_numeric(target_vals, errors='coerce')
                if numeric_vals.notna().sum() / len(target_vals) >= SchemaDetector.NUMERIC_THRESHOLD:
                    schema['target'] = {
                        'name': target_col,
                        'type': 'numeric',
                        'min': float(numeric_vals.min()),
                        'max': float(numeric_vals.max())
                    }
                else:
                    # fallback categorical anyway
                    schema['target'] = {
                        'name': target_col,
                        'type': 'categorical',
                        'cardinality': int(cardinality),
                        'class_distribution': target_vals.value_counts().to_dict()
                    }
        
        return schema


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
