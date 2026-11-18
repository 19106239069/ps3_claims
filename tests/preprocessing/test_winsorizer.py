
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from ps3.preprocessing import Winsorizer


@pytest.mark.parametrize(
    "lower_quantile, upper_quantile",
    [(0, 1), (0.05, 0.95), (0.5, 0.5)],
)
def test_winsorizer(lower_quantile, upper_quantile):
    """Test Winsorizer with different quantile settings."""

    # Create a deterministic dataset
    X = np.arange(1, 1001, dtype=float)

    # Fit and transform the data
    winsorizer = Winsorizer(
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    X_transformed = winsorizer.fit_transform(X)

    # Calculate the expected quantiles
    expected_lower = np.quantile(X, lower_quantile)
    expected_upper = np.quantile(X, upper_quantile)

    # Check that the fitted quantiles are correct
    assert winsorizer.lower_quantile_ == expected_lower
    assert winsorizer.upper_quantile_ == expected_upper

    # Check that transformed values are within the boundaries
    assert np.all(X_transformed >= expected_lower)
    assert np.all(X_transformed <= expected_upper)

    # Check that the output has the same shape as the input
    assert X_transformed.shape == X.shape


def test_winsorizer_2d():
    """Test that each feature is winsorized independently."""

    X = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
        [100, 400],
    ], dtype=float)

    winsorizer = Winsorizer(
        lower_quantile=0.2,
        upper_quantile=0.8,
    )

    X_transformed = winsorizer.fit_transform(X)

    # Calculate the expected quantiles for each column
    expected_lower = np.quantile(X, 0.2, axis=0)
    expected_upper = np.quantile(X, 0.8, axis=0)

    expected = np.clip(X, expected_lower, expected_upper)

    np.testing.assert_allclose(X_transformed, expected)


def test_transform_before_fit():
    """Test that transform raises an error before fitting."""

    winsorizer = Winsorizer()

    with pytest.raises(NotFittedError):
        winsorizer.transform(np.array([1, 2, 3]))


@pytest.mark.parametrize(
    "lower_quantile, upper_quantile",
    [
        (-0.1, 0.95),
        (0.05, 1.1),
        (0.9, 0.1),
    ],
)
def test_invalid_quantiles(lower_quantile, upper_quantile):
    """Test that invalid quantiles raise a ValueError."""

    X = np.array([1, 2, 3, 4, 5])

    winsorizer = Winsorizer(
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )

    with pytest.raises(ValueError):
        winsorizer.fit(X)
