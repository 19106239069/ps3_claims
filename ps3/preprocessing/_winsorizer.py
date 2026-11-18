
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Winsorizer(BaseEstimator, TransformerMixin):

    def __init__(self, lower_quantile=0.05, upper_quantile=0.95):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X = np.asarray(X)

        # Check the input dimensions
        if X.ndim not in (1, 2):
            raise ValueError("X must be a 1D or 2D array.")

        # Check that the quantiles are valid
        if not 0 <= self.lower_quantile <= self.upper_quantile <= 1:
            raise ValueError(
                "Quantiles must satisfy 0 <= lower_quantile "
                "<= upper_quantile <= 1."
            )

        # Calculate and store the lower and upper quantiles
        self.lower_quantile_ = np.quantile(
            X, self.lower_quantile, axis=0
        )
        self.upper_quantile_ = np.quantile(
            X, self.upper_quantile, axis=0
        )

        return self

    def transform(self, X):
        # Check whether the transformer has been fitted
        check_is_fitted(
            self, attributes=["lower_quantile_", "upper_quantile_"]
        )

        X = np.asarray(X)

        # Clip values outside the fitted quantile boundaries
        return np.clip(
            X,
            self.lower_quantile_,
            self.upper_quantile_
        )
