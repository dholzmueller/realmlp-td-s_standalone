import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import TensorDataset, DataLoader

from preprocessing import get_realmlp_td_s_pipeline


class ScalingLayer(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale[None, :]


class NTPLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, zero_init: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factor = 0.0 if zero_init else 1.0
        self.weight = nn.Parameter(factor * torch.randn(in_features, out_features))
        self.bias = nn.Parameter(factor * torch.randn(1, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (1. / np.sqrt(self.in_features)) * (x @ self.weight) + self.bias


class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul(torch.tanh(torch.nn.functional.softplus(x)))


class SimpleMLP(BaseEstimator):
    def __init__(self, is_classification: bool, device: str = 'cpu'):
        self.is_classification = is_classification
        self.device = device

    def fit(self, X, y, X_val=None, y_val=None):
        # print(f'fit {X=}')
        input_dim = X.shape[1]
        is_classification = self.is_classification

        output_dim = 1 if len(y.shape) == 1 else y.shape[1]

        if self.is_classification:
            self.class_enc_ = OrdinalEncoder(dtype=np.int64)
            y = self.class_enc_.fit_transform(y[:, None])[:, 0]
            self.classes_ = self.class_enc_.categories_[0]
            output_dim = len(self.class_enc_.categories_[0])
        else:  # standardize targets
            self.y_mean_ = np.mean(y, axis=0)
            self.y_std_ = np.std(y, axis=0)
            y = (y - self.y_mean_) / (self.y_std_ + 1e-30)
            if y_val is not None:
                y_val = (y_val - self.y_mean_) / (self.y_std_ + 1e-30)

        act = nn.SELU if is_classification else Mish
        model = nn.Sequential(
            ScalingLayer(input_dim),
            NTPLinear(input_dim, 256), act(),
            NTPLinear(256, 256), act(),
            NTPLinear(256, 256), act(),
            NTPLinear(256, output_dim, zero_init=True),
        ).to(self.device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1) if is_classification else nn.MSELoss()
        params = list(model.parameters())
        scale_params = [params[0]]
        weights = params[1::2]
        biases = params[2::2]
        opt = torch.optim.Adam([dict(params=scale_params), dict(params=weights), dict(params=biases)],
                                betas=(0.9, 0.95))

        x_train = torch.as_tensor(X, dtype=torch.float32)
        y_train = torch.as_tensor(y, dtype=torch.int64 if self.is_classification else torch.float32)
        if not is_classification and len(y_train.shape) == 1:
            y_train = y_train[:, None]

        if X_val is not None and y_val is not None:
            x_valid = torch.as_tensor(X_val, dtype=torch.float32)
            y_valid = torch.as_tensor(y_val, dtype=torch.int64 if self.is_classification else torch.float32)
            if not is_classification and len(y_valid.shape) == 1:
                y_valid = y_valid[:, None]
        else:
            x_valid = x_train[:0]
            y_valid = y_train[:0]

        train_ds = TensorDataset(x_train, y_train)
        valid_ds = TensorDataset(x_valid, y_valid)
        n_train = x_train.shape[0]
        n_valid = x_valid.shape[0]
        n_epochs = 256
        train_batch_size = min(256, n_train)
        valid_batch_size = max(1, min(1024, n_valid))

        def valid_metric(y_pred: torch.Tensor, y: torch.Tensor):
            if self.is_classification:
                # unnormalized classification error, could also convert to float and then take the mean
                return torch.sum(torch.argmax(y_pred, dim=-1) != y)
            else:
                # MSE
                return (y_pred - y).square().mean()

        train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, drop_last=True)
        valid_dl = DataLoader(valid_ds, batch_size=valid_batch_size, shuffle=False)

        n_train_batches = len(train_dl)
        base_lr = 0.04 if is_classification else 0.07
        best_valid_loss = np.Inf
        best_valid_params = None

        for epoch in range(n_epochs):
            # print(f'Epoch {epoch + 1}/{n_epochs}')
            for batch_idx, (x_batch, y_batch) in enumerate(train_dl):
                # set learning rates according to schedule
                t = (epoch * n_train_batches + batch_idx) / (n_epochs * n_train_batches)
                lr_sched_value = 0.5 - 0.5 * np.cos(2 * np.pi * np.log2(1 + 15 * t))
                lr = base_lr * lr_sched_value
                # print(f'{lr=:g}')
                opt.param_groups[0]['lr'] = 6 * lr  # for scale
                opt.param_groups[1]['lr'] = lr  # for weights
                opt.param_groups[2]['lr'] = 0.1 * lr  # for biases

                # optimization
                y_pred = model(x_batch.to(self.device))
                loss = criterion(y_pred, y_batch.to(self.device))
                loss.backward()
                opt.step()
                opt.zero_grad()
                # print(f'{loss.item()=:g}')

            # save parameters if validation score improves
            with torch.no_grad():
                if x_valid.shape[0] > 0.0:
                    y_pred_valid = torch.cat([model(x_batch.to(self.device)).detach() for x_batch, _ in valid_dl], dim=0)
                    valid_loss = valid_metric(y_pred_valid, y_valid.to(self.device)).cpu().item()
                else:
                    valid_loss = 0.0
                if valid_loss <= best_valid_loss:  # use <= for last best epoch
                    best_valid_loss = valid_loss
                    best_valid_params = [p.detach().clone() for p in model.parameters()]

        # after training, revert to best epoch
        with torch.no_grad():
            for p_model, p_copy in zip(model.parameters(), best_valid_params):
                p_model.set_(p_copy)

        self.model_ = model

        return self

    def predict(self, X):
        x = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            y_pred = self.model_(x).cpu().numpy()
        if self.is_classification:
            # return classes with highest probability
            return self.class_enc_.inverse_transform(np.argmax(y_pred, axis=-1)[:, None])[:, 0]
        else:
            return y_pred[:, 0] * self.y_std_ + self.y_mean_

    def predict_proba(self, X):
        assert self.is_classification
        self.model_.eval()
        x = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y_pred = torch.softmax(self.model_(x), dim=-1).cpu().numpy()
        return y_pred


class Standalone_RealMLP_TD_S_Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, device: str = 'cpu'):
        self.device = device

    def fit(self, X, y, X_val=None, y_val=None):
        self.prep_ = get_realmlp_td_s_pipeline()
        self.model_ = SimpleMLP(is_classification=True, device=self.device)
        X = self.prep_.fit_transform(X)
        if X_val is not None:
            X_val = self.prep_.transform(X_val)
        self.model_.fit(X, y, X_val, y_val)
        self.classes_ = self.model_.classes_

    def predict(self, X):
        return self.model_.predict(self.prep_.transform(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self.prep_.transform(X))


class Standalone_RealMLP_TD_S_Regressor(BaseEstimator, RegressorMixin):
    def __init__(self, device: str = 'cpu'):
        self.device = device

    def fit(self, X, y, X_val=None, y_val=None):
        self.prep_ = get_realmlp_td_s_pipeline()
        self.model_ = SimpleMLP(is_classification=False, device=self.device)
        X = self.prep_.fit_transform(X)
        if X_val is not None:
            X_val = self.prep_.transform(X_val)
        self.model_.fit(X, y, X_val, y_val)

    def predict(self, X):
        return self.model_.predict(self.prep_.transform(X))
