from urllib import request

from fastapi import FastAPI, UploadFile, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import io
import uuid
import logging
from data_handler import SchemaDetector
from ctgan_wrapper import CTGANWrapper
from evaluation.quality import QualityMetrics
from evaluation.privacy import PrivacyMetrics
from downstream.classifier import ClassifierPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Synthetic Data Generator", version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

datasets = {}
models = {}

class TrainRequest(BaseModel):
    dataset_id: str
    epochs: int = 100

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}

@app.post("/api/upload")
async def upload_csv(
    file: UploadFile = File(...),
    target: str = Form('fraud')
):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    
    schema = SchemaDetector.detect(df, target_col=target)

    if not schema.get('target'):
        raise ValueError(f"Target column '{target}' not found or invalid")
    
    dataset_id = str(uuid.uuid4())
    datasets[dataset_id] = {
        'df': df,
        'schema': schema
    }
    
    return {
        'dataset_id': dataset_id,
        'n_rows': len(df),
        'n_features': len(schema['features']),
        'class_dist': schema['target']['class_distribution'],
        'schema': schema
    }

@app.post("/api/train-ctgan")
async def train_ctgan(request: TrainRequest):
    df = datasets[request.dataset_id]['df']
    schema = datasets[request.dataset_id]['schema']

    ctgan = CTGANWrapper(schema, epochs=request.epochs)
    ctgan.train(df, target_col=schema['target']['name'])
    
    model_path = f"/models/ctgan_{request.dataset_id}.pkl"
    ctgan.save(model_path)
    
    return {'model_id': request.dataset_id, 'status': 'trained'}

@app.post("/api/generate")
async def generate_synthetic(dataset_id: str, n_samples: int = 5000):
    # Load model
    ctgan = CTGANWrapper.load(f"/models/ctgan_{dataset_id}.pkl")
    
    # Generate
    synthetic_df = ctgan.generate(n_samples)
    
    # Store
    syn_id = str(uuid.uuid4())
    datasets[syn_id] = {'df': synthetic_df, 'schema': datasets[dataset_id]['schema']}
    
    return {
        'synthetic_id': syn_id,
        'n_samples': len(synthetic_df),
        'preview': synthetic_df.head(5).to_dict(orient='records')
    }

@app.post("/api/evaluate")
async def evaluate(dataset_id: str, synthetic_id: str):
    real_df = datasets[dataset_id]['df']
    syn_df = datasets[synthetic_id]['df']
    
    quality = {
        'kl_divergence': QualityMetrics.kl_divergence(real_df, syn_df),
        'wasserstein': QualityMetrics.wasserstein_distance(real_df, syn_df),
        'correlation_diff': QualityMetrics.correlation_difference(real_df, syn_df),
        'pca_variance': QualityMetrics.pca_variance_retained(real_df, syn_df)
    }
    
    privacy = {
        'k_anonymity': PrivacyMetrics.k_anonymity(real_df, syn_df),
        'mia_auc': PrivacyMetrics.membership_inference_attack(real_df, syn_df),
        'dp_estimate': PrivacyMetrics.dp_budget_estimate(real_df, syn_df)
    }
    
    # Classify
    classifier = ClassifierPipeline(datasets[dataset_id]['schema'], target_col=datasets[dataset_id]['schema']['target']['name'])
    _, real_metrics = classifier.train_real_only(real_df)
    _, mixed_metrics = classifier.train_synthetic_mixed(real_df, syn_df, synthetic_weight=0.5)
    
    return {
        'quality': quality,
        'privacy': privacy,
        'real_only_metrics': real_metrics,
        'synthetic_mixed_metrics': mixed_metrics,
        'impact': {
            'auc_lift': (mixed_metrics['auc'] - real_metrics['auc']) / real_metrics['auc'],
            'recall_lift': (mixed_metrics['recall_minority'] - real_metrics['recall_minority']) / real_metrics['recall_minority']
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)