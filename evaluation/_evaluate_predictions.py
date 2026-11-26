
import numpy as np
import pandas as pd
from glum import TweedieDistribution
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error


def evaluate_predictions(y_true, y_pred, sample_weight):
    """
    Calculate performance metrics for model predictions.

    Parameters
    ----------
    y_true : array-like
        Actual outcome values (pure premiums).
    y_pred : array-like
        Predicted outcome values.
    sample_weight : array-like
        Sample weights, represented by exposure.

    Returns
    -------
    pd.DataFrame
        DataFrame containing Bias, Deviance, MAE, RMSE and Gini.
        Metric names are used as the index.
    """

    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)

    # Validate inputs
    if not (y_true.ndim == y_pred.ndim == sample_weight.ndim == 1):
        raise ValueError("All inputs must be one-dimensional.")

    if not (len(y_true) == len(y_pred) == len(sample_weight)):
        raise ValueError("Inputs must have the same length.")

    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty.")

    if not (
        np.isfinite(y_true).all()
        and np.isfinite(y_pred).all()
        and np.isfinite(sample_weight).all()
    ):
        raise ValueError("Inputs must contain only finite values.")

    if np.any(sample_weight < 0) or sample_weight.sum() <= 0:
        raise ValueError("Sample weights must be non-negative and sum to a positive value.")

    # 1. Calculate exposure-weighted Bias
    actual_mean = np.average(
        y_true,
        weights=sample_weight
    )

    predicted_mean = np.average(
        y_pred,
        weights=sample_weight
    )

    bias = predicted_mean - actual_mean

    # 2. Calculate exposure-weighted Tweedie Deviance
    tweedie_dist = TweedieDistribution(1.5)

    deviance = (
        tweedie_dist.deviance(
            y_true,
            y_pred,
            sample_weight=sample_weight
        )
        / sample_weight.sum()
    )

    # 3. Calculate exposure-weighted MAE
    mae = mean_absolute_error(
        y_true,
        y_pred,
        sample_weight=sample_weight
    )

    # 4. Calculate exposure-weighted RMSE
    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
            sample_weight=sample_weight
        )
    )

    # 5. Calculate Gini coefficient using the Lorenz curve
    # Sort observations by increasing predicted risk
    ranking = np.argsort(y_pred)

    ranked_exposure = sample_weight[ranking]
    ranked_pure_premium = y_true[ranking]

    # Cumulative claim amount
    cumulated_claim_amount = np.cumsum(
        ranked_pure_premium * ranked_exposure
    )

    total_claim_amount = cumulated_claim_amount[-1]

    if total_claim_amount <= 0:
        gini = np.nan
    else:
        cumulated_claim_amount /= total_claim_amount

        cumulated_samples = np.linspace(
            0,
            1,
            len(cumulated_claim_amount)
        )

        gini = 1 - 2 * auc(
            cumulated_samples,
            cumulated_claim_amount
        )

    # Return metrics as a DataFrame
    metrics = {
        "Bias": bias,
        "Deviance": deviance,
        "MAE": mae,
        "RMSE": rmse,
        "Gini": gini
    }

    return pd.DataFrame.from_dict(
        metrics,
        orient="index",
        columns=["Value"]
    )
