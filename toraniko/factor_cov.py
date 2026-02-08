"""Barra-style factor covariance matrix and specific risk estimation.

Implements:
1. Exponentially-weighted covariance matrix (EWMA)
2. Newey-West autocorrelation adjustment
3. Eigenvalue adjustment (floor clipping or Ledoit-Wolf shrinkage)
4. Specific (idiosyncratic) risk estimation

Usage:
    factor_cov, factor_names, spec_risk = estimate_factor_cov(
        factor_returns=factor_returns,
        residual_returns=residual_returns,
        window=252,
        half_life=90,
        max_lag=2,
    )
"""

import numpy as np
import polars as pl

from toraniko.factor_math import exp_weights


def _extract_factor_matrix(
    factor_returns: pl.DataFrame,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Extract the (T x K) numpy factor return matrix from a Polars DataFrame.

    Sorts by date ascending. Returns the matrix, ordered factor names,
    and the sorted date array.

    Parameters
    ----------
    factor_returns : pl.DataFrame
        Factor returns DataFrame with a 'date' column and K factor columns.

    Returns
    -------
    F : np.ndarray, shape (T, K)
        Factor return matrix sorted by date.
    factor_names : list[str]
        Ordered factor column names corresponding to columns of F.
    dates : np.ndarray
        Sorted date values.

    Raises
    ------
    TypeError
        If factor_returns is not a Polars DataFrame.
    ValueError
        If 'date' column is missing or there are fewer than 2 observations.
    """
    if not isinstance(factor_returns, pl.DataFrame):
        raise TypeError(f"Expected pl.DataFrame, got {type(factor_returns)}")
    if "date" not in factor_returns.columns:
        raise ValueError("factor_returns must have a 'date' column")

    df = factor_returns.sort("date")
    factor_names = [c for c in df.columns if c != "date"]

    if len(factor_names) == 0:
        raise ValueError("factor_returns has no factor columns (only 'date')")
    if df.height < 2:
        raise ValueError(f"Need at least 2 observations, got {df.height}")

    F = df.select(factor_names).to_numpy()
    dates = df["date"].to_numpy()

    # Check for NaN/Inf
    if not np.isfinite(F).all():
        raise ValueError("factor_returns contains NaN or Inf values")

    return F, factor_names, dates


def ewma_cov(
    factor_returns: pl.DataFrame,
    window: int = 252,
    half_life: int = 90,
) -> tuple[np.ndarray, list[str]]:
    """Compute the exponentially-weighted covariance matrix of factor returns.

    Uses the trailing ``window`` observations with exponential decay at
    the given ``half_life``.

    Parameters
    ----------
    factor_returns : pl.DataFrame
        Output of ``estimate_factor_returns()``. Must have 'date' column
        plus K factor columns.
    window : int
        Number of trailing trading days to use. If the DataFrame has fewer
        rows, all available data is used.
    half_life : int
        Half-life for exponential decay in trading days.

    Returns
    -------
    cov : np.ndarray, shape (K, K)
        Exponentially-weighted covariance matrix.
    factor_names : list[str]
        Factor names corresponding to rows/columns of the matrix.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if half_life < 1:
        raise ValueError(f"half_life must be >= 1, got {half_life}")

    F, factor_names, _ = _extract_factor_matrix(factor_returns)

    T_available = F.shape[0]
    T_use = min(window, T_available)
    F_use = F[-T_use:]  # most recent observations

    w = exp_weights(T_use, half_life)  # shape (T_use,), chronological order
    W_bar = w.sum()

    # Weighted mean: (K,)
    mu = (w @ F_use) / W_bar

    # De-mean: (T_use, K)
    F_tilde = F_use - mu[np.newaxis, :]

    # Efficient weighted covariance: F_w = F_tilde * sqrt(w), then cov = F_w^T F_w / W_bar
    F_w = F_tilde * np.sqrt(w)[:, np.newaxis]
    cov = (F_w.T @ F_w) / W_bar

    # Ensure perfect symmetry (eliminate floating-point drift)
    cov = (cov + cov.T) / 2

    return cov, factor_names


def newey_west_adjust(
    factor_returns: pl.DataFrame,
    cov_base: np.ndarray,
    window: int = 252,
    half_life: int = 90,
    max_lag: int = 2,
) -> np.ndarray:
    """Apply Newey-West autocorrelation adjustment to a base covariance matrix.

    Adds Bartlett-weighted lagged cross-autocovariance terms to correct
    for serial correlation in daily factor returns.

    Parameters
    ----------
    factor_returns : pl.DataFrame
        Factor returns DataFrame (same as used for ``ewma_cov``).
    cov_base : np.ndarray
        The (K, K) exponentially-weighted covariance matrix from ``ewma_cov()``.
    window : int
        Number of trailing observations (should match the ``ewma_cov`` call).
    half_life : int
        Half-life for exponential weighting (should match the ``ewma_cov`` call).
    max_lag : int
        Maximum lag for the Newey-West correction. Barra uses 2 for daily data.
        Use 0 to skip the adjustment (returns cov_base unchanged).

    Returns
    -------
    cov_nw : np.ndarray, shape (K, K)
        Newey-West adjusted covariance matrix.
    """
    if max_lag < 0:
        raise ValueError(f"max_lag must be >= 0, got {max_lag}")
    if max_lag == 0:
        return cov_base.copy()

    F, _, _ = _extract_factor_matrix(factor_returns)

    T_available = F.shape[0]
    T_use = min(window, T_available)
    F_use = F[-T_use:]

    w = exp_weights(T_use, half_life)
    W_bar = w.sum()

    # De-mean using weighted means (same as ewma_cov)
    mu = (w @ F_use) / W_bar
    F_tilde = F_use - mu[np.newaxis, :]

    cov_nw = cov_base.copy()

    for d in range(1, max_lag + 1):
        if d >= T_use:
            break  # not enough data for this lag

        # Bartlett taper weight
        k_d = 1.0 - d / (max_lag + 1)

        # Lag-d cross-autocovariance: Gamma_d = sum_t w_t * F_tilde_t * F_tilde_{t-d}^T / W_bar
        F_lead = F_tilde[d:]       # (T_use - d, K), observations at time t
        F_lag = F_tilde[:-d]       # (T_use - d, K), observations at time t-d
        w_d = w[d:]                # weights for the lead observations

        # Efficient computation: weight the lead, then multiply
        F_lead_w = F_lead * w_d[:, np.newaxis]  # (T_use - d, K)
        Gamma_d = (F_lead_w.T @ F_lag) / W_bar  # (K, K)

        # Add symmetrized lagged autocovariance
        cov_nw += k_d * (Gamma_d + Gamma_d.T)

    # Ensure symmetry
    cov_nw = (cov_nw + cov_nw.T) / 2

    return cov_nw


def eigen_adjust(
    cov_matrix: np.ndarray,
    method: str = "floor",
    floor_ratio: float = 1e-4,
) -> np.ndarray:
    """Apply eigenvalue adjustment to ensure positive definiteness.

    Parameters
    ----------
    cov_matrix : np.ndarray
        The (K, K) covariance matrix (typically after Newey-West adjustment).
    method : str
        ``"floor"`` — clip eigenvalues at ``max_eigenvalue * floor_ratio``.
        ``"shrinkage"`` — Ledoit-Wolf shrinkage toward scaled identity.
    floor_ratio : float
        For method="floor": minimum eigenvalue as a fraction of the largest.

    Returns
    -------
    cov_adj : np.ndarray, shape (K, K)
        Adjusted positive-definite covariance matrix.
    """
    if method not in ("floor", "shrinkage"):
        raise ValueError(f"method must be 'floor' or 'shrinkage', got '{method}'")

    K = cov_matrix.shape[0]

    if method == "floor":
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        floor_val = eigenvalues.max() * floor_ratio
        eigenvalues_adj = np.maximum(eigenvalues, floor_val)
        cov_adj = eigenvectors @ np.diag(eigenvalues_adj) @ eigenvectors.T

    elif method == "shrinkage":
        # Ledoit-Wolf shrinkage toward scaled identity
        # Target: (trace(S) / K) * I
        mu_target = np.trace(cov_matrix) / K
        target = mu_target * np.eye(K)

        # Optimal shrinkage intensity (Oracle Approximating Shrinkage, simplified)
        # Uses the Ledoit-Wolf (2004) formula for shrinkage toward scaled identity
        # alpha = min(1, max(0, estimated_optimal_alpha))
        delta = cov_matrix - target
        delta_sq_sum = np.sum(delta ** 2)

        # For the simplified case, use a heuristic based on the ratio of
        # off-diagonal to total variance. This avoids needing the raw data.
        # A more precise implementation would use the raw factor returns.
        off_diag_mask = ~np.eye(K, dtype=bool)
        off_diag_sq = np.sum(cov_matrix[off_diag_mask] ** 2)
        diag_sq = np.sum(np.diag(cov_matrix) ** 2)
        total_sq = off_diag_sq + diag_sq

        if total_sq > 0:
            # Shrink more when off-diagonal elements are large relative to diagonal
            alpha = min(1.0, max(0.0, off_diag_sq / total_sq))
        else:
            alpha = 0.0

        cov_adj = (1 - alpha) * cov_matrix + alpha * target

    # Ensure symmetry
    cov_adj = (cov_adj + cov_adj.T) / 2

    return cov_adj


def specific_risk(
    residual_returns: pl.DataFrame,
    window: int = 252,
    half_life: int = 90,
) -> pl.DataFrame:
    """Compute exponentially-weighted idiosyncratic (specific) variances.

    Parameters
    ----------
    residual_returns : pl.DataFrame
        Residual returns from ``estimate_factor_returns()``.
        Has 'date' column plus one column per stock symbol.
    window : int
        Number of trailing observations for the variance estimate.
    half_life : int
        Half-life for exponential decay.

    Returns
    -------
    pl.DataFrame
        Columns: ``symbol``, ``specific_variance``, ``specific_vol``.
        One row per stock, sorted by symbol.
    """
    if not isinstance(residual_returns, pl.DataFrame):
        raise TypeError(f"Expected pl.DataFrame, got {type(residual_returns)}")
    if "date" not in residual_returns.columns:
        raise ValueError("residual_returns must have a 'date' column")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if half_life < 1:
        raise ValueError(f"half_life must be >= 1, got {half_life}")

    df = residual_returns.sort("date")
    stock_cols = [c for c in df.columns if c != "date"]

    if len(stock_cols) == 0:
        raise ValueError("residual_returns has no stock columns (only 'date')")

    eps = df.select(stock_cols).to_numpy()  # (T, N)

    T_available = eps.shape[0]
    T_use = min(window, T_available)
    eps_use = eps[-T_use:]

    w = exp_weights(T_use, half_life)
    W_bar = w.sum()

    # Weighted mean
    mu_eps = (w @ eps_use) / W_bar  # (N,)

    # De-mean
    eps_tilde = eps_use - mu_eps[np.newaxis, :]

    # Weighted variance per stock
    spec_var = (w @ (eps_tilde ** 2)) / W_bar  # (N,)

    return pl.DataFrame({
        "symbol": stock_cols,
        "specific_variance": spec_var,
        "specific_vol": np.sqrt(spec_var),
    }).sort("symbol")


def estimate_factor_cov(
    factor_returns: pl.DataFrame,
    residual_returns: pl.DataFrame,
    window: int = 252,
    half_life: int = 90,
    max_lag: int = 2,
    eigen_method: str = "floor",
    floor_ratio: float = 1e-4,
) -> tuple[np.ndarray, list[str], pl.DataFrame]:
    """Estimate full Barra-style factor covariance matrix and specific risk.

    Chains: ewma_cov → newey_west_adjust → eigen_adjust, plus specific_risk.

    Parameters
    ----------
    factor_returns : pl.DataFrame
        Factor returns from ``estimate_factor_returns()``.
    residual_returns : pl.DataFrame
        Residual returns from ``estimate_factor_returns()``.
    window : int
        Number of trailing observations. Default 252 (~1 year).
    half_life : int
        Exponential decay half-life in trading days. Default 90.
    max_lag : int
        Newey-West maximum lag. Default 2.
    eigen_method : str
        Eigenvalue adjustment: ``"floor"`` or ``"shrinkage"``.
    floor_ratio : float
        Floor ratio for eigenvalue clipping.

    Returns
    -------
    cov_adj : np.ndarray, shape (K, K)
        Adjusted factor covariance matrix (positive definite).
    factor_names : list[str]
        Ordered factor names corresponding to matrix rows/columns.
    spec_risk_df : pl.DataFrame
        Per-stock specific risk (symbol, specific_variance, specific_vol).
    """
    # Step 1: Exponentially-weighted covariance
    cov_base, factor_names = ewma_cov(factor_returns, window=window, half_life=half_life)

    # Step 2: Newey-West autocorrelation adjustment
    cov_nw = newey_west_adjust(
        factor_returns, cov_base, window=window, half_life=half_life, max_lag=max_lag
    )

    # Step 3: Eigenvalue adjustment
    cov_adj = eigen_adjust(cov_nw, method=eigen_method, floor_ratio=floor_ratio)

    # Step 4: Specific (idiosyncratic) risk
    spec_risk_df = specific_risk(residual_returns, window=window, half_life=half_life)

    return cov_adj, factor_names, spec_risk_df
