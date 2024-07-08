import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.ordinal_enc_ = OrdinalEncoder(unknown_value=np.NaN, encoded_missing_value=np.NaN,
                                           handle_unknown='use_encoded_value')
        self.ordinal_enc_.fit(X)
        self.cat_sizes_ = []
        for cat_arr in self.ordinal_enc_.categories_:
            has_nan = np.any([isinstance(val, (float, np.float32, np.float64)) and np.isnan(val) for val in cat_arr])
            self.cat_sizes_.append(len(cat_arr) - int(has_nan))

        return self

    def transform(self, X, y=None):
        # print(f'transform {X=}')
        x_enc = self.ordinal_enc_.transform(X)
        n_samples = x_enc.shape[0]
        out_arrs = []
        for i, cat_size in enumerate(self.cat_sizes_):
            column = x_enc[:, i]
            idxs = np.arange(n_samples)
            isnan = np.isnan(column)
            out_arr = np.zeros(shape=(n_samples, cat_size))
            # do one-hot encoding, encode nan (missing or unknown) values to all zeros
            out_arr[idxs[~isnan], column[~isnan].astype(np.int64)] = 1.

            if cat_size == 2:
                # binary: encode to single feature being -1, 1 or 0 (for missing or unknown values)
                out_arr = out_arr[:, 0:1] - out_arr[:, 1:2]

            out_arrs.append(out_arr)

        return np.concatenate(out_arrs, axis=-1)


class CustomOneHotPipeline(BaseEstimator, TransformerMixin):
    """
    Apply CustomOneHotEncoder only to categorical features.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.tfm_ = ColumnTransformer(transformers=[
            ('categorical', CustomOneHotEncoder(),
             make_column_selector(dtype_include=["string", "object", "category"])),
            ('remaining', FunctionTransformer(lambda x: x),
             make_column_selector(dtype_exclude=["string", "object", "category"]))
        ]).fit(X)
        return self

    def transform(self, X, y=None):
        # print(f'{X=}')
        X = pd.DataFrame(X)
        # print(f'{X=}')
        return self.tfm_.transform(X)


class RobustScaleSmoothClipTransform(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, X, y=None):
        # don't deal with dataframes for simplicity
        assert isinstance(X, np.ndarray)
        self._median = np.median(X, axis=-2)
        quant_diff = np.quantile(X, 0.75, axis=-2) - np.quantile(X, 0.25, axis=-2)
        max = np.max(X, axis=-2)
        min = np.min(X, axis=-2)
        idxs = quant_diff == 0.0
        # on indexes where the quantile difference is zero, do min-max scaling instead
        quant_diff[idxs] = 0.5 * (max[idxs] - min[idxs])
        factors = 1.0 / (quant_diff + 1e-30)
        # if feature is constant on the training data,
        # set factor to zero so that it is also constant at prediction time
        factors[quant_diff == 0.0] = 0.0
        self._factors = factors
        return self

    def transform(self, X, y=None):
        x_scaled = self._factors[None, :] * (X - self._median[None, :])
        return x_scaled / np.sqrt(1 + (x_scaled / 3) ** 2)


def get_realmlp_td_s_pipeline():
    return sklearn.pipeline.Pipeline([('one_hot', CustomOneHotPipeline()),
                                      ('rssc', RobustScaleSmoothClipTransform())])
