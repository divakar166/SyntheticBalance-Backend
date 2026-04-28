from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pickle
import torch
from ctgan import CTGAN
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.ctgan import Discriminator, Generator
from torch import optim

from handlers.data_handler import Preprocessor, normalize_dataframe
from settings import get_settings

ProgressCallback = Callable[[int, int, Dict[str, float]], None]


def _sdmetrics_report(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, metadata: dict) -> dict:
    try:
        from sdmetrics.reports.single_table import DiagnosticReport, QualityReport
        from sdmetrics.single_table import BinaryAdaBoostClassifier
    except ImportError:
        return {"error": "sdmetrics not installed – pip install sdmetrics"}

    results: dict = {}
    meta_cols = list(metadata.get("columns", {}).keys())
    real_df = real_df[[c for c in meta_cols if c in real_df.columns]]
    synthetic_df = synthetic_df[[c for c in meta_cols if c in synthetic_df.columns]]

    try:
        quality_report = QualityReport()
        quality_report.generate(real_df, synthetic_df, metadata, verbose=False)
        results["quality_score"] = round(float(quality_report.get_score()), 4)

        property_scores = {}
        for prop_name in quality_report.get_properties()["Property"].tolist():
            prop_score = quality_report.get_properties()
            row = prop_score[prop_score["Property"] == prop_name]
            if not row.empty:
                property_scores[prop_name] = round(float(row["Score"].iloc[0]), 4)
        results["quality_properties"] = property_scores

        try:
            col_shapes = quality_report.get_details(property_name="Column Shapes")
            if col_shapes is not None and not col_shapes.empty:
                results["column_shape_scores"] = {
                    row["Column"]: round(float(row["Score"]), 4)
                    for _, row in col_shapes.iterrows()
                    if pd.notna(row.get("Score"))
                }
        except Exception:
            pass

        try:
            pair_trends = quality_report.get_details(property_name="Column Pair Trends")
            if pair_trends is not None and not pair_trends.empty:
                results["column_pair_trend_scores"] = [
                    {
                        "column_1": row["Column 1"],
                        "column_2": row["Column 2"],
                        "score": round(float(row["Score"]), 4),
                        "metric": row.get("Metric", ""),
                    }
                    for _, row in pair_trends.iterrows()
                    if pd.notna(row.get("Score"))
                ]
        except Exception:
            pass

    except Exception as exc:
        results["quality_error"] = str(exc)

    try:
        diag_report = DiagnosticReport()
        diag_report.generate(real_df, synthetic_df, metadata, verbose=False)
        results["diagnostic_score"] = round(float(diag_report.get_score()), 4)

        diag_properties = {}
        for prop_name in diag_report.get_properties()["Property"].tolist():
            prop_score = diag_report.get_properties()
            row = prop_score[prop_score["Property"] == prop_name]
            if not row.empty:
                diag_properties[prop_name] = round(float(row["Score"].iloc[0]), 4)
        results["diagnostic_properties"] = diag_properties

    except Exception as exc:
        results["diagnostic_error"] = str(exc)

    try:
        target_col = metadata.get("_target_col")
        if target_col and target_col in real_df.columns:
            real_encoded = real_df.apply(
                lambda s: s.astype('category').cat.codes if s.dtype == object else s
            )
            synth_encoded = synthetic_df.apply(
                lambda s: s.astype('category').cat.codes if s.dtype == object else s
            )
            ml_result = BinaryAdaBoostClassifier.compute(
                test_data=real_encoded,
                train_data=synth_encoded,
                target=target_col,
                metadata=metadata,
            )
            results["ml_efficacy"] = {
                "train_on_synthetic_test_on_real_f1": round(float(ml_result), 4)
            }
    except Exception as exc:
        results["ml_efficacy_error"] = str(exc)

    return results


def build_sdmetrics_metadata(schema: dict, target_col: str | None = None) -> dict:
    columns: dict = {}
    for col_name, col_meta in schema.get("features", {}).items():
        if col_meta.get("type") == "numeric":
            columns[col_name] = {"sdtype": "numerical"}
        else:
            columns[col_name] = {"sdtype": "categorical"}

    target_meta = schema.get("target")
    if target_meta and target_col:
        ttype = target_meta.get("type", "categorical")
        columns[target_col] = {"sdtype": "numerical" if ttype == "numeric" else "categorical"}

    metadata = {"columns": columns}
    if target_col:
        metadata["_target_col"] = target_col
    return metadata


def _get_pac(model: CTGAN) -> int:
    return getattr(model, 'pac', None) or getattr(model, '_pac', 1)


def _round_to_pac(n: int, pac: int) -> int:
    return max(pac, int(np.ceil(n / pac)) * pac)


def _mode_collapse_score(loss_history: List[Dict], window: int = 10) -> float:
    if len(loss_history) < window:
        return -1.0
    recent = [x["generator_loss"] for x in loss_history[-window:]]
    return float(np.std(recent))


def _detect_convergence(loss_history: List[Dict], patience: int, min_delta: float) -> bool:
    if len(loss_history) < patience:
        return False
    recent = [x["generator_loss"] for x in loss_history[-patience:]]
    return (max(recent) - min(recent)) < min_delta


class CTGANWrapper:
    def __init__(
        self,
        schema: Dict,
        epochs: int | None = None,
        batch_size: int | None = None,
        # User-configurable training params
        generator_lr: float | None = None,
        discriminator_lr: float | None = None,
        discriminator_steps: int | None = None,
        embedding_dim: int | None = None,
        generator_dim: Tuple[int, ...] | None = None,
        discriminator_dim: Tuple[int, ...] | None = None,
        # Early stopping
        early_stopping: bool = True,
        early_stopping_patience: int = 20,
        early_stopping_min_delta: float = 0.001,
    ):
        settings = get_settings()
        self.schema = schema
        self.epochs = epochs if epochs is not None else settings.ctgan_epochs_default
        self.pac = 1

        requested_batch_size = batch_size if batch_size is not None else settings.ctgan_batch_size
        self.batch_size = _round_to_pac(requested_batch_size, self.pac)

        self.generator_lr = generator_lr if generator_lr is not None else 2e-4
        self.discriminator_lr = discriminator_lr if discriminator_lr is not None else 2e-4
        self.discriminator_steps = discriminator_steps if discriminator_steps is not None else 1
        self.embedding_dim = embedding_dim if embedding_dim is not None else 128
        self.generator_dim = generator_dim if generator_dim is not None else (256, 256)
        self.discriminator_dim = discriminator_dim if discriminator_dim is not None else (256, 256)

        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        self.model = None
        self.preprocessor = None
        self.training_history: List[Dict] = []
        self.training_time_seconds = 0.0
        self.convergence_epoch: int | None = None

        self.sdmetrics_report: dict | None = None

    def get_config(self) -> Dict:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "generator_lr": self.generator_lr,
            "discriminator_lr": self.discriminator_lr,
            "discriminator_steps": self.discriminator_steps,
            "embedding_dim": self.embedding_dim,
            "generator_dim": list(self.generator_dim),
            "discriminator_dim": list(self.discriminator_dim),
            "early_stopping": self.early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
        }
    

    def evaluate_quality(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        target_col: str | None = None,
    ) -> dict:
        skipped = self.schema.get('skipped_features', [])
        real_df_eval = real_df.drop(columns=[c for c in skipped if c in real_df.columns]).copy()

        for col, meta in self.schema['features'].items():
            if meta['type'] == 'categorical' and col in real_df_eval.columns:
                if pd.api.types.is_numeric_dtype(real_df_eval[col]):
                    real_df_eval[col] = real_df_eval[col].astype(str)

        metadata = build_sdmetrics_metadata(self.schema, target_col=target_col)
        return _sdmetrics_report(real_df_eval, synthetic_df, metadata)


    def train(
        self,
        df: pd.DataFrame,
        target_col: str = None,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        self._current_target_col = target_col
        self.preprocessor = Preprocessor(self.schema)
        self.preprocessor.fit(df)
        train_df = self._prepare_training_data(df, target_col=target_col)
        categorical_cols = self._get_discrete_columns(target_col=target_col)

        print(f"Training CTGAN for {self.epochs} epochs on {len(df)} samples...")
        print(f"Config: {self.get_config()}")

        self.model = CTGAN(
            epochs=1,
            batch_size=self.batch_size,
            verbose=False,
            pac=self.pac,
            embedding_dim=self.embedding_dim,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            generator_lr=self.generator_lr,
            discriminator_lr=self.discriminator_lr,
            discriminator_steps=self.discriminator_steps,
        )
        self.training_history = []
        self.convergence_epoch = None
        self.training_time_seconds = self._fit_with_progress(
            train_data=train_df,
            discrete_columns=categorical_cols,
            progress_callback=progress_callback,
        )

    def _get_discrete_columns(self, target_col: str | None = None) -> list[str]:
        categorical_cols = [
            col for col, meta in self.schema['features'].items()
            if meta['type'] == 'categorical'
        ]
        if (
            target_col
            and self.schema.get('target')
            and self.schema['target'].get('type') == 'categorical'
            and target_col not in categorical_cols
        ):
            categorical_cols.append(target_col)
        return categorical_cols


    def _prepare_training_data(self, df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
        train_df = normalize_dataframe(df)
        if target_col and target_col in train_df.columns:
            train_df = train_df[train_df[target_col].notna()].copy()
            if train_df.empty:
                raise ValueError(f"Target column '{target_col}' has no usable non-missing rows for training.")
        for col, meta in self.schema['features'].items():
            if col not in train_df.columns:
                continue
            if meta['type'] == 'numeric':
                numeric_series = pd.to_numeric(train_df[col], errors='coerce')
                invalid_mask = train_df[col].notna() & numeric_series.isna()
                if invalid_mask.any():
                    raise ValueError(f"Column '{col}' is marked numeric but contains non-numeric values.")
                fill_value = numeric_series.median()
                if pd.isna(fill_value):
                    fill_value = 0.0
                train_df[col] = numeric_series.fillna(fill_value).astype('float64')
            else:
                train_df[col] = (
                    train_df[col]
                    .astype(str)
                    .replace("<NA>", "__missing__")
                    .replace("nan", "__missing__")
                    .astype(object)
                )
        if target_col and target_col in train_df.columns:
            target_meta = self.schema.get('target') or {}
            if target_meta.get('type') == 'categorical':
                # train_df[target_col] = train_df[target_col].astype('object')
                train_df[target_col] = (
                    train_df[target_col]
                    .astype(str)
                    .replace({"<NA>": "__missing__", "nan": "__missing__"})
                    .astype(object)
                )
            elif target_meta.get('type') == 'numeric':
                numeric_target = pd.to_numeric(train_df[target_col], errors='coerce')
                invalid_mask = train_df[target_col].notna() & numeric_target.isna()
                if invalid_mask.any():
                    raise ValueError(f"Target column '{target_col}' is marked numeric but contains non-numeric values.")
                train_df[target_col] = numeric_target.astype('float64')
        return train_df


    def _fit_with_progress(
        self,
        train_data: pd.DataFrame,
        discrete_columns,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> float:
        model = self.model
        pac = _get_pac(model)
        batch_size = _round_to_pac(model._batch_size, pac)
        if batch_size != model._batch_size:
            model._batch_size = batch_size

        model._validate_discrete_columns(train_data, discrete_columns)
        model._validate_null_data(train_data, discrete_columns)
        
        for col in train_data.columns:
            if col in discrete_columns:
                train_data[col] = train_data[col].astype(str).astype(object)
            else:
                col_meta = self.schema.get('features', {}).get(col, {})
                target_meta = self.schema.get('target', {})
                is_numeric_in_schema = (
                    col_meta.get('type') == 'numeric' or
                    (col == getattr(self, '_current_target_col', None) and target_meta.get('type') == 'numeric')
                )
                if is_numeric_in_schema:
                    train_data[col] = pd.to_numeric(train_data[col], errors='coerce').astype('float64')
                else:
                    train_data[col] = train_data[col].astype(str).astype(object)
                    if col not in discrete_columns:
                        discrete_columns.append(col)

        model._transformer = DataTransformer()
        model._transformer.fit(train_data, discrete_columns)
        transformed_data = model._transformer.transform(train_data)
        model._data_sampler = DataSampler(
            transformed_data, model._transformer.output_info_list, model._log_frequency
        )
        data_dim = model._transformer.output_dimensions

        model._generator = Generator(
            model._embedding_dim + model._data_sampler.dim_cond_vec(),
            model._generator_dim,
            data_dim,
        ).to(model._device)

        discriminator = Discriminator(
            data_dim + model._data_sampler.dim_cond_vec(),
            model._discriminator_dim,
            pac=pac,
        ).to(model._device)

        optimizer_g = optim.Adam(
            model._generator.parameters(),
            lr=model._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=model._generator_decay,
        )
        optimizer_d = optim.Adam(
            discriminator.parameters(),
            lr=model._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=model._discriminator_decay,
        )

        mean = torch.zeros(batch_size, model._embedding_dim, device=model._device)
        std = mean + 1
        model.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Discriminator Loss'])

        start = perf_counter()
        steps_per_epoch = max(len(transformed_data) // batch_size, 1)
        n_rows = len(train_data)

        for epoch_index in range(self.epochs):
            epoch_start = perf_counter()
            last_generator_loss = None
            last_discriminator_loss = None

            for step in range(steps_per_epoch):
                batch_index = step + 1
                for _ in range(model._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)
                    condvec = model._data_sampler.sample_condvec(batch_size)
                    if condvec is None:
                        c1, col, opt = None, None, None
                        real = model._data_sampler.sample_data(transformed_data, batch_size, col, opt)
                    else:
                        c1, _m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(model._device)
                        fakez = torch.cat([fakez, c1], dim=1)
                        perm = np.arange(batch_size)
                        np.random.shuffle(perm)
                        real = model._data_sampler.sample_data(transformed_data, batch_size, col[perm], opt[perm])
                        c2 = c1[perm]
                    fake = model._generator(fakez)
                    fakeact = model._apply_activate(fake)
                    real = torch.from_numpy(real.astype('float32')).to(model._device)
                    if condvec is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        fake_cat = fakeact
                        real_cat = real
                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)
                    pen = discriminator.calc_gradient_penalty(real_cat, fake_cat, model._device, pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                    last_discriminator_loss = float(loss_d.detach().cpu().item())
                    optimizer_d.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizer_d.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = model._data_sampler.sample_condvec(batch_size)
                if condvec is None:
                    c1, m1 = None, None
                else:
                    c1, m1, _col, _opt = condvec
                    c1 = torch.from_numpy(c1).to(model._device)
                    m1 = torch.from_numpy(m1).to(model._device)
                    fakez = torch.cat([fakez, c1], dim=1)
                fake = model._generator(fakez)
                fakeact = model._apply_activate(fake)
                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                    cross_entropy = model._cond_loss(fake, c1, m1)
                else:
                    y_fake = discriminator(fakeact)
                    cross_entropy = 0
                loss_g = -torch.mean(y_fake) + cross_entropy
                last_generator_loss = float(loss_g.detach().cpu().item())
                optimizer_g.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizer_g.step()

                if progress_callback:
                    progress_callback(
                        epoch_index + 1,
                        self.epochs,
                        {
                            'stage': 'batch',
                            'current_batch': batch_index,
                            'total_batches': steps_per_epoch,
                            'generator_loss': last_generator_loss,
                            'discriminator_loss': last_discriminator_loss,
                        },
                    )

            epoch_time = perf_counter() - epoch_start
            generator_loss = float(loss_g.detach().cpu().item())
            discriminator_loss = float(loss_d.detach().cpu().item())

            loss_ratio = abs(generator_loss) / (abs(discriminator_loss) + 1e-8)

            epoch_record = {
                'epoch': epoch_index + 1,
                'generator_loss': generator_loss,
                'discriminator_loss': discriminator_loss,
                'loss_ratio': round(loss_ratio, 4),
                'epoch_time_seconds': round(epoch_time, 3),
                'samples_per_second': round(n_rows / epoch_time, 1),
            }
            self.training_history.append(epoch_record)

            mcs = _mode_collapse_score(self.training_history)
            if mcs >= 0:
                epoch_record['mode_collapse_score'] = round(mcs, 6)

            epoch_loss_df = pd.DataFrame({
                'Epoch': [epoch_index],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
            })
            model.loss_values = (
                epoch_loss_df if model.loss_values.empty
                else pd.concat([model.loss_values, epoch_loss_df]).reset_index(drop=True)
            )

            if progress_callback:
                progress_callback(
                    epoch_index + 1,
                    self.epochs,
                    {**epoch_record, 'stage': 'epoch', 'current_batch': steps_per_epoch, 'total_batches': steps_per_epoch},
                )

            if self.early_stopping and _detect_convergence(
                self.training_history,
                self.early_stopping_patience,
                self.early_stopping_min_delta,
            ):
                self.convergence_epoch = epoch_index + 1
                print(
                    f"[Early Stop] Converged at epoch {self.convergence_epoch}/{self.epochs} "
                    f"(patience={self.early_stopping_patience}, min_delta={self.early_stopping_min_delta})"
                )
                break

        return perf_counter() - start

    def generate(self, n_samples: int, condition: Dict[str, int] = None) -> pd.DataFrame:
        if condition:
            col, val = list(condition.items())[0]
            return self.model.sample(n_samples, condition_column=col, condition_value=val)
        return self.model.sample(n_samples)


    def save(self, path: str):
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'model': self.model,
            'schema': self.schema,
            'preprocessor_scaler': self.preprocessor.numeric_scaler,
            'preprocessor_encoder': self.preprocessor.categorical_encoder,
            'training_history': self.training_history,
            'training_time_seconds': self.training_time_seconds,
            'convergence_epoch': self.convergence_epoch,
            'config': self.get_config(),
            'sdmetrics_report': self.sdmetrics_report,
        }
        with model_path.open('wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {model_path}")

    @classmethod
    def load(cls, path: str) -> 'CTGANWrapper':
        with Path(path).open('rb') as f:
            data = pickle.load(f)
        cfg = data.get('config', {})
        wrapper = cls(
            schema=data['schema'],
            epochs=cfg.get('epochs'),
            batch_size=cfg.get('batch_size'),
            generator_lr=cfg.get('generator_lr'),
            discriminator_lr=cfg.get('discriminator_lr'),
            discriminator_steps=cfg.get('discriminator_steps'),
            embedding_dim=cfg.get('embedding_dim'),
            generator_dim=tuple(cfg['generator_dim']) if cfg.get('generator_dim') else None,
            discriminator_dim=tuple(cfg['discriminator_dim']) if cfg.get('discriminator_dim') else None,
        )
        wrapper.model = data['model']
        wrapper.preprocessor = Preprocessor(data['schema'])
        wrapper.preprocessor.numeric_scaler = data['preprocessor_scaler']
        wrapper.preprocessor.categorical_encoder = data['preprocessor_encoder']
        wrapper.preprocessor.fitted = True
        wrapper.training_history = data.get('training_history', [])
        wrapper.training_time_seconds = data.get('training_time_seconds', 0.0)
        wrapper.convergence_epoch = data.get('convergence_epoch')
        wrapper.sdmetrics_report = data.get('sdmetrics_report')
        return wrapper