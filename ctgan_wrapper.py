from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Optional

import pandas as pd
import numpy as np
import pickle
from backend.handlers.data_handler import Preprocessor
from ctgan import CTGAN
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.ctgan import Discriminator, Generator
import torch
from torch import optim


ProgressCallback = Callable[[int, int, Dict[str, float]], None]

class CTGANWrapper:
    def __init__(self, schema: Dict, epochs: int = 100, batch_size: int = 256):
        self.schema = schema
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.preprocessor = None
        self.training_history = []
        self.training_time_seconds = 0.0

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = None,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        """
        Train CTGAN on real data.
        
        Args:
            df: Real training data
            target_col: Optional; if provided, use class-conditional generation
            progress_callback: Optional callback called after each epoch
        """
        # Preprocess
        self.preprocessor = Preprocessor(self.schema)
        self.preprocessor.fit(df)

        categorical_cols = [
            col for col, meta in self.schema['features'].items()
            if meta['type'] == 'categorical'
        ]

        print(f"Training CTGAN for {self.epochs} epochs on {len(df)} samples...")

        self.model = CTGAN(
            epochs=1,
            batch_size=self.batch_size,
            verbose=False
        )

        self.training_history = []
        self.training_time_seconds = self._fit_with_progress(
            train_data=df,
            discrete_columns=categorical_cols,
            progress_callback=progress_callback,
        )

    def _fit_with_progress(
        self,
        train_data: pd.DataFrame,
        discrete_columns,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> float:
        model = self.model
        model._validate_discrete_columns(train_data, discrete_columns)
        model._validate_null_data(train_data, discrete_columns)

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
            pac=model.pac,
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

        mean = torch.zeros(model._batch_size, model._embedding_dim, device=model._device)
        std = mean + 1
        model.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

        start = perf_counter()
        steps_per_epoch = max(len(transformed_data) // model._batch_size, 1)
        for epoch_index in range(self.epochs):
            for _ in range(steps_per_epoch):
                for _ in range(model._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)
                    condvec = model._data_sampler.sample_condvec(model._batch_size)
                    if condvec is None:
                        c1, col, opt = None, None, None
                        real = model._data_sampler.sample_data(
                            transformed_data,
                            model._batch_size,
                            col,
                            opt,
                        )
                    else:
                        c1, _m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(model._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(model._batch_size)
                        np.random.shuffle(perm)
                        real = model._data_sampler.sample_data(
                            transformed_data,
                            model._batch_size,
                            col[perm],
                            opt[perm],
                        )
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
                    pen = discriminator.calc_gradient_penalty(
                        real_cat,
                        fake_cat,
                        model._device,
                        model.pac,
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizer_d.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizer_d.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = model._data_sampler.sample_condvec(model._batch_size)

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

                optimizer_g.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizer_g.step()

            generator_loss = float(loss_g.detach().cpu().item())
            discriminator_loss = float(loss_d.detach().cpu().item())
            epoch_record = {
                'epoch': epoch_index + 1,
                'generator_loss': generator_loss,
                'discriminator_loss': discriminator_loss,
            }
            self.training_history.append(epoch_record)

            epoch_loss_df = pd.DataFrame(
                {
                    'Epoch': [epoch_index],
                    'Generator Loss': [generator_loss],
                    'Distriminator Loss': [discriminator_loss],
                }
            )
            if model.loss_values.empty:
                model.loss_values = epoch_loss_df
            else:
                model.loss_values = pd.concat([model.loss_values, epoch_loss_df]).reset_index(drop=True)

            if progress_callback:
                progress_callback(epoch_index + 1, self.epochs, epoch_record)

        return perf_counter() - start
        
    def generate(self, n_samples: int, condition: Dict[str, int] = None) -> pd.DataFrame:
        """Sample synthetic data from trained model"""        
        if condition:
            col, val = list(condition.items())[0]
            return self.model.sample(n_samples, condition_column=col, condition_value=val)
        
        return self.model.sample(n_samples)
    
    def save(self, path: str):
        """Serialize model to disk"""
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'model': self.model,
            'schema': self.schema,
            'preprocessor_scaler': self.preprocessor.numeric_scaler,
            'preprocessor_encoder': self.preprocessor.categorical_encoder,
            'training_history': self.training_history,
            'training_time_seconds': self.training_time_seconds,
        }
        with model_path.open('wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {model_path}")
    
    @classmethod
    def load(cls, path: str) -> 'CTGANWrapper':
        """Deserialize model from disk"""
        with Path(path).open('rb') as f:
            data = pickle.load(f)
        
        wrapper = cls(schema=data['schema'])
        wrapper.model = data['model']
        wrapper.preprocessor = Preprocessor(data['schema'])
        wrapper.preprocessor.numeric_scaler = data['preprocessor_scaler']
        wrapper.preprocessor.categorical_encoder = data['preprocessor_encoder']
        wrapper.preprocessor.fitted = True
        wrapper.training_history = data.get('training_history', [])
        wrapper.training_time_seconds = data.get('training_time_seconds', 0.0)
        
        return wrapper
