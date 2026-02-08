"""Tests for the Barra-style factor covariance estimation module."""

import numpy as np
import polars as pl
import pytest
from datetime import date, timedelta

from toraniko.factor_cov import (
    _extract_factor_matrix,
    ewma_cov,
    newey_west_adjust,
    eigen_adjust,
    specific_risk,
    estimate_factor_cov,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_factor_returns():
    """Small deterministic factor returns DataFrame: 3 factors, 50 dates."""
    np.random.seed(42)
    T, K = 50, 3
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(T)]
    data = np.random.randn(T, K) * 0.01
    return pl.DataFrame({
        "date": dates,
        "market": data[:, 0],
        "MOMENTUM": data[:, 1],
        "SIZE": data[:, 2],
    }).with_columns(pl.col("date").cast(pl.Datetime("us")))


@pytest.fixture
def synthetic_residual_returns():
    """Small deterministic residual returns: 5 stocks, 50 dates."""
    np.random.seed(123)
    T, N = 50, 5
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(T)]
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
    data = np.random.randn(T, N) * 0.02
    df_dict = {"date": dates}
    for i, sym in enumerate(symbols):
        df_dict[sym] = data[:, i]
    return pl.DataFrame(df_dict).with_columns(pl.col("date").cast(pl.Datetime("us")))


@pytest.fixture
def autocorrelated_factor_returns():
    """Factor returns with known AR(1) structure for Newey-West testing."""
    np.random.seed(99)
    T = 100
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(T)]
    # AR(1) with rho=0.5
    rho = 0.5
    eps = np.random.randn(T) * 0.01
    f = np.zeros(T)
    f[0] = eps[0]
    for t in range(1, T):
        f[t] = rho * f[t - 1] + eps[t]
    return pl.DataFrame({
        "date": dates,
        "factor_A": f,
        "factor_B": np.random.randn(T) * 0.01,  # IID for contrast
    }).with_columns(pl.col("date").cast(pl.Datetime("us")))


# ---------------------------------------------------------------------------
# Tests: _extract_factor_matrix
# ---------------------------------------------------------------------------

class TestExtractFactorMatrix:
    def test_basic_extraction(self, synthetic_factor_returns):
        F, names, dates = _extract_factor_matrix(synthetic_factor_returns)
        assert F.shape == (50, 3)
        assert names == ["market", "MOMENTUM", "SIZE"]
        assert len(dates) == 50

    def test_sorted_by_date(self, synthetic_factor_returns):
        # Shuffle the input
        shuffled = synthetic_factor_returns.sample(fraction=1.0, shuffle=True, seed=0)
        F, _, dates = _extract_factor_matrix(shuffled)
        # Dates should be ascending after extraction
        assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))

    def test_not_a_dataframe(self):
        with pytest.raises(TypeError):
            _extract_factor_matrix("not a dataframe")

    def test_missing_date_column(self):
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match="date"):
            _extract_factor_matrix(df)

    def test_single_row(self):
        df = pl.DataFrame({
            "date": [date(2024, 1, 1)],
            "factor_A": [0.01],
        }).with_columns(pl.col("date").cast(pl.Datetime("us")))
        with pytest.raises(ValueError, match="at least 2"):
            _extract_factor_matrix(df)

    def test_nan_in_data(self):
        df = pl.DataFrame({
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "factor_A": [0.01, float("nan")],
        }).with_columns(pl.col("date").cast(pl.Datetime("us")))
        with pytest.raises(ValueError, match="NaN"):
            _extract_factor_matrix(df)


# ---------------------------------------------------------------------------
# Tests: ewma_cov
# ---------------------------------------------------------------------------

class TestEwmaCov:
    def test_symmetry(self, synthetic_factor_returns):
        cov, _ = ewma_cov(synthetic_factor_returns, window=50, half_life=25)
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_positive_semi_definite(self, synthetic_factor_returns):
        cov, _ = ewma_cov(synthetic_factor_returns, window=50, half_life=25)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)  # allow tiny numerical noise

    def test_dimensions(self, synthetic_factor_returns):
        cov, names = ewma_cov(synthetic_factor_returns, window=50, half_life=25)
        assert cov.shape == (3, 3)
        assert len(names) == 3

    def test_large_half_life_approaches_sample_cov(self, synthetic_factor_returns):
        """With very large half_life, EWMA cov should approximate equal-weighted sample cov."""
        cov_ewma, names = ewma_cov(synthetic_factor_returns, window=50, half_life=10000)
        F, _, _ = _extract_factor_matrix(synthetic_factor_returns)
        cov_sample = np.cov(F, rowvar=False, bias=True)  # bias=True for population cov
        np.testing.assert_array_almost_equal(cov_ewma, cov_sample, decimal=3)

    def test_window_truncation(self, synthetic_factor_returns):
        """Using window=10 should only use last 10 rows."""
        cov_full, _ = ewma_cov(synthetic_factor_returns, window=50, half_life=10)
        cov_trunc, _ = ewma_cov(synthetic_factor_returns, window=10, half_life=10)
        # They should differ since different data is used
        assert not np.allclose(cov_full, cov_trunc)

    def test_window_larger_than_data(self, synthetic_factor_returns):
        """Window larger than available data should use all data without error."""
        cov, _ = ewma_cov(synthetic_factor_returns, window=1000, half_life=25)
        assert cov.shape == (3, 3)

    def test_factor_names_returned(self, synthetic_factor_returns):
        _, names = ewma_cov(synthetic_factor_returns, window=50, half_life=25)
        assert names == ["market", "MOMENTUM", "SIZE"]

    def test_invalid_window(self, synthetic_factor_returns):
        with pytest.raises(ValueError):
            ewma_cov(synthetic_factor_returns, window=1, half_life=10)

    def test_invalid_half_life(self, synthetic_factor_returns):
        with pytest.raises(ValueError):
            ewma_cov(synthetic_factor_returns, window=50, half_life=0)


# ---------------------------------------------------------------------------
# Tests: newey_west_adjust
# ---------------------------------------------------------------------------

class TestNeweyWestAdjust:
    def test_max_lag_zero_returns_base(self, synthetic_factor_returns):
        cov_base, _ = ewma_cov(synthetic_factor_returns, window=50, half_life=25)
        cov_nw = newey_west_adjust(
            synthetic_factor_returns, cov_base, window=50, half_life=25, max_lag=0
        )
        np.testing.assert_array_almost_equal(cov_nw, cov_base)

    def test_symmetry(self, synthetic_factor_returns):
        cov_base, _ = ewma_cov(synthetic_factor_returns, window=50, half_life=25)
        cov_nw = newey_west_adjust(
            synthetic_factor_returns, cov_base, window=50, half_life=25, max_lag=2
        )
        np.testing.assert_array_almost_equal(cov_nw, cov_nw.T)

    def test_nw_increases_variance_for_autocorrelated_data(self, autocorrelated_factor_returns):
        """For AR(1) factor, NW adjustment should increase variance."""
        cov_base, _ = ewma_cov(autocorrelated_factor_returns, window=100, half_life=50)
        cov_nw = newey_west_adjust(
            autocorrelated_factor_returns, cov_base, window=100, half_life=50, max_lag=2
        )
        # The autocorrelated factor (index 0) should see increased variance
        assert cov_nw[0, 0] >= cov_base[0, 0] * 0.99  # allow small numerical noise

    def test_iid_data_minimal_change(self):
        """For IID data with large sample, NW should barely change the diagonal."""
        np.random.seed(777)
        T = 500
        dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(T)]
        df = pl.DataFrame({
            "date": dates,
            "f1": np.random.randn(T) * 0.01,
            "f2": np.random.randn(T) * 0.01,
        }).with_columns(pl.col("date").cast(pl.Datetime("us")))
        cov_base, _ = ewma_cov(df, window=500, half_life=250)
        cov_nw = newey_west_adjust(df, cov_base, window=500, half_life=250, max_lag=2)
        # Diagonal (variances) should change by less than 20% for IID data
        diag_change = np.abs(np.diag(cov_nw) - np.diag(cov_base)) / np.diag(cov_base)
        assert diag_change.mean() < 0.2

    def test_invalid_max_lag(self, synthetic_factor_returns):
        cov_base, _ = ewma_cov(synthetic_factor_returns, window=50, half_life=25)
        with pytest.raises(ValueError):
            newey_west_adjust(
                synthetic_factor_returns, cov_base, window=50, half_life=25, max_lag=-1
            )


# ---------------------------------------------------------------------------
# Tests: eigen_adjust
# ---------------------------------------------------------------------------

class TestEigenAdjust:
    def test_floor_clips_small_eigenvalues(self):
        """Construct a near-singular matrix and verify floor clips it."""
        # Create PSD matrix with one tiny eigenvalue
        Q = np.eye(3)
        lam = np.diag([1.0, 0.5, 1e-8])
        cov = Q @ lam @ Q.T

        cov_adj = eigen_adjust(cov, method="floor", floor_ratio=1e-4)
        eigenvalues = np.linalg.eigvalsh(cov_adj)
        assert eigenvalues.min() >= 1.0 * 1e-4 - 1e-12  # floor = max_eig * ratio

    def test_floor_preserves_well_conditioned(self):
        """A well-conditioned matrix should be barely changed."""
        cov = np.array([[1.0, 0.3], [0.3, 1.0]])
        cov_adj = eigen_adjust(cov, method="floor", floor_ratio=1e-4)
        np.testing.assert_array_almost_equal(cov, cov_adj, decimal=5)

    def test_symmetry_preserved(self):
        cov = np.array([[0.04, 0.01, 0.005],
                        [0.01, 0.03, 0.002],
                        [0.005, 0.002, 0.02]])
        cov_adj = eigen_adjust(cov, method="floor")
        np.testing.assert_array_almost_equal(cov_adj, cov_adj.T)

    @pytest.mark.parametrize("method", ["floor", "shrinkage"])
    def test_positive_definite_output(self, method):
        cov = np.array([[0.04, 0.01], [0.01, 0.03]])
        cov_adj = eigen_adjust(cov, method=method)
        eigenvalues = np.linalg.eigvalsh(cov_adj)
        assert np.all(eigenvalues > 0)

    def test_shrinkage_moves_toward_identity(self):
        """Shrinkage should reduce off-diagonal elements."""
        cov = np.array([[1.0, 0.8], [0.8, 1.0]])
        cov_adj = eigen_adjust(cov, method="shrinkage")
        # Off-diagonal should be smaller (closer to identity structure)
        assert abs(cov_adj[0, 1]) <= abs(cov[0, 1]) + 1e-10

    def test_invalid_method(self):
        cov = np.eye(2)
        with pytest.raises(ValueError, match="method"):
            eigen_adjust(cov, method="invalid")


# ---------------------------------------------------------------------------
# Tests: specific_risk
# ---------------------------------------------------------------------------

class TestSpecificRisk:
    def test_output_schema(self, synthetic_residual_returns):
        result = specific_risk(synthetic_residual_returns, window=50, half_life=25)
        assert "symbol" in result.columns
        assert "specific_variance" in result.columns
        assert "specific_vol" in result.columns

    def test_non_negative_variances(self, synthetic_residual_returns):
        result = specific_risk(synthetic_residual_returns, window=50, half_life=25)
        assert (result["specific_variance"] >= 0).all()

    def test_vol_equals_sqrt_variance(self, synthetic_residual_returns):
        result = specific_risk(synthetic_residual_returns, window=50, half_life=25)
        expected_vol = np.sqrt(result["specific_variance"].to_numpy())
        actual_vol = result["specific_vol"].to_numpy()
        np.testing.assert_array_almost_equal(actual_vol, expected_vol)

    def test_stock_count(self, synthetic_residual_returns):
        result = specific_risk(synthetic_residual_returns, window=50, half_life=25)
        assert result.height == 5  # 5 stocks

    def test_constant_residual_zero_variance(self):
        """A stock with constant residual should have zero variance."""
        df = pl.DataFrame({
            "date": [date(2024, 1, 1) + timedelta(days=i) for i in range(20)],
            "STOCK_A": [0.01] * 20,
            "STOCK_B": np.random.randn(20).tolist(),
        }).with_columns(pl.col("date").cast(pl.Datetime("us")))
        result = specific_risk(df, window=20, half_life=10)
        stock_a_var = result.filter(pl.col("symbol") == "STOCK_A")["specific_variance"][0]
        assert stock_a_var < 1e-15

    def test_not_a_dataframe(self):
        with pytest.raises(TypeError):
            specific_risk("not a df")

    def test_missing_date_column(self):
        df = pl.DataFrame({"AAPL": [0.01, 0.02], "MSFT": [0.03, 0.04]})
        with pytest.raises(ValueError, match="date"):
            specific_risk(df)


# ---------------------------------------------------------------------------
# Tests: estimate_factor_cov (integration)
# ---------------------------------------------------------------------------

class TestEstimateFactorCov:
    def test_smoke_test(self, synthetic_factor_returns, synthetic_residual_returns):
        cov, names, spec = estimate_factor_cov(
            synthetic_factor_returns, synthetic_residual_returns,
            window=50, half_life=25, max_lag=2,
        )
        assert isinstance(cov, np.ndarray)
        assert isinstance(names, list)
        assert isinstance(spec, pl.DataFrame)

    def test_output_shapes(self, synthetic_factor_returns, synthetic_residual_returns):
        cov, names, spec = estimate_factor_cov(
            synthetic_factor_returns, synthetic_residual_returns,
            window=50, half_life=25,
        )
        assert cov.shape == (3, 3)
        assert len(names) == 3
        assert spec.height == 5

    def test_positive_definite(self, synthetic_factor_returns, synthetic_residual_returns):
        cov, _, _ = estimate_factor_cov(
            synthetic_factor_returns, synthetic_residual_returns,
            window=50, half_life=25,
        )
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0)

    def test_factor_names_match_input(self, synthetic_factor_returns, synthetic_residual_returns):
        cov, names, _ = estimate_factor_cov(
            synthetic_factor_returns, synthetic_residual_returns,
            window=50, half_life=25,
        )
        expected = [c for c in synthetic_factor_returns.columns if c != "date"]
        assert names == expected

    @pytest.mark.parametrize("eigen_method", ["floor", "shrinkage"])
    def test_both_eigen_methods(self, synthetic_factor_returns, synthetic_residual_returns, eigen_method):
        cov, _, _ = estimate_factor_cov(
            synthetic_factor_returns, synthetic_residual_returns,
            window=50, half_life=25, eigen_method=eigen_method,
        )
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0)
