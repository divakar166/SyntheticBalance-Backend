from typing import Dict
import pandas as pd
import numpy as np
import pickle
from data_handler import Preprocessor
from ctgan import CTGAN

class CTGANWrapper:
    def __init__(self, schema: Dict, epochs: int = 100, batch_size: int = 256):
        self.schema = schema
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.preprocessor = None
        self.training_history = []
        
    def train(self, df: pd.DataFrame, target_col: str = None):
        """
        Train CTGAN on real data.
        
        Args:
            df: Real training data
            target_col: Optional; if provided, use class-conditional generation
        """
        # Preprocess
        self.preprocessor = Preprocessor(self.schema)
        self.preprocessor.fit(df)
        df_proc = self.preprocessor.transform(df)

        categorical_cols = [
            col for col, meta in self.schema['features'].items()
            if meta['type'] == 'categorical'
        ]
        
        print(f"Training CTGAN for {self.epochs} epochs on {len(df)} samples...")
        
        self.model = CTGAN(
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=True
        )
        
        self.model.fit(df, discrete_columns=categorical_cols)
        
    def generate(self, n_samples: int, condition: Dict[str, int] = None) -> pd.DataFrame:
        """Sample synthetic data from trained model"""        
        if condition:
            col, val = list(condition.items())[0]
            return self.model.sample(n_samples, condition_column=col, condition_value=val)
        
        return self.model.sample(n_samples)
    
    def save(self, path: str):
        """Serialize model to disk"""
        data = {
            'model': self.model,
            'schema': self.schema,
            'preprocessor_scaler': self.preprocessor.numeric_scaler,
            'preprocessor_encoder': self.preprocessor.categorical_encoder,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'CTGANWrapper':
        """Deserialize model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        wrapper = cls(schema=data['schema'])
        wrapper.model = data['model']
        wrapper.preprocessor = Preprocessor(data['schema'])
        wrapper.preprocessor.numeric_scaler = data['preprocessor_scaler']
        wrapper.preprocessor.categorical_encoder = data['preprocessor_encoder']
        wrapper.preprocessor.fitted = True
        
        return wrapper
