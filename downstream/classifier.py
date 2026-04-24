from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np

class ClassifierPipeline:
    """Train XGBoost on real vs synthetic+real data"""
    
    def __init__(self, schema: Dict, random_seed: int = 42):
        self.schema = schema
        self.seed = random_seed
        self.target_col = schema['target']['name'] if schema['target'] else None
        
    def train_real_only(self, df_real: pd.DataFrame) -> Tuple[Any, Dict]:
        """
        Train on real data only (baseline).
        """
        try:
            import xgboost as xgb
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix
        except ImportError:
            print("XGBoost not installed. Use: pip install xgboost scikit-learn")
            return None, {}
        
        y = df_real[self.target_col]
        X = df_real.drop(columns=[self.target_col])
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed, stratify=y
        )
        
        # Handle class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Train
        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            max_depth=6,
            n_estimators=100,
            random_state=self.seed,
            eval_metric='logloss',
            verbosity=0
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'auc': roc_auc_score(y_test, y_proba),
            'f1': f1_score(y_test, y_pred),
            'recall_minority': recall_score(y_test, y_pred),
            'accuracy': (y_pred == y_test).mean(),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return model, metrics
    
    def train_synthetic_mixed(
        self, 
        df_real: pd.DataFrame, 
        df_synthetic: pd.DataFrame, 
        synthetic_weight: float = 0.5
    ) -> Tuple[Any, Dict]:
        """
        Train on real + synthetic blended.
        
        Args:
            df_real: Real training data
            df_synthetic: Synthetic data (from generative model)
            synthetic_weight: How much to emphasize synthetic samples (0-1)
        """
        try:
            import xgboost as xgb
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix
        except ImportError:
            return None, {}
        
        y_real = df_real[self.target_col]
        X_real = df_real.drop(columns=[self.target_col])
        
        y_syn = df_synthetic[self.target_col]
        X_syn = df_synthetic.drop(columns=[self.target_col])
        
        # Combine
        X_combined = pd.concat([X_real, X_syn], ignore_index=True)
        y_combined = pd.concat([y_real, y_syn], ignore_index=True)
        
        # Sample weights: real=1.0, synthetic=synthetic_weight
        sample_weights = np.array(
            [1.0] * len(df_real) + [synthetic_weight] * len(df_synthetic)
        )
        
        # Train/test split (on combined data)
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X_combined, y_combined, sample_weights,
            test_size=0.2, random_state=self.seed, stratify=y_combined
        )
        
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            max_depth=6,
            n_estimators=100,
            random_state=self.seed,
            eval_metric='logloss',
            verbosity=0
        )
        model.fit(X_train, y_train, sample_weight=w_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'auc': roc_auc_score(y_test, y_proba),
            'f1': f1_score(y_test, y_pred),
            'recall_minority': recall_score(y_test, y_pred),
            'accuracy': (y_pred == y_test).mean(),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return model, metrics
