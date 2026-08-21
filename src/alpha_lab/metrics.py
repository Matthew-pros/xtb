from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew


def performance(returns: pd.Series, annualization: int = 365) -> dict[str, float]:
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 2:
        raise ValueError("Insufficient returns")
    wealth = (1 + r).cumprod()
    years = len(r) / annualization
    cagr = wealth.iloc[-1] ** (1 / years) - 1 if wealth.iloc[-1] > 0 else -1.0
    vol = r.std(ddof=1) * math.sqrt(annualization)
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * math.sqrt(annualization)
    dd = wealth / wealth.cummax() - 1
    maxdd = float(dd.min())
    sharpe = (
        r.mean() / r.std(ddof=1) * math.sqrt(annualization) if r.std(ddof=1) else np.nan
    )
    return {
        "observations": len(r),
        "years": years,
        "total_return": wealth.iloc[-1] - 1,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": cagr / downside if downside else np.nan,
        "max_drawdown": maxdd,
        "calmar": cagr / abs(maxdd) if maxdd else np.nan,
        "skew": r.skew(),
        "worst_day": r.min(),
        "hit_rate": (r > 0).mean(),
    }


def probabilistic_sharpe(
    returns: pd.Series, benchmark_sr: float = 0.0, annualization: int = 365
) -> float:
    """Bailey/Lopez de Prado PSR, corrected for skew and excess kurtosis."""
    r = pd.Series(returns).dropna().to_numpy()
    n = len(r)
    sr = np.mean(r) / np.std(r, ddof=1)
    sr0 = benchmark_sr / math.sqrt(annualization)
    g3 = skew(r, bias=False)
    g4 = kurtosis(r, fisher=False, bias=False)
    denom = math.sqrt(max(1 - g3 * sr + ((g4 - 1) / 4) * sr * sr, 1e-12))
    return float(norm.cdf((sr - sr0) * math.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected maximum under independent trials; disclosed conservative approximation."""
    if n_trials <= 1:
        return 0.0
    euler = 0.5772156649
    z1 = norm.ppf(1 - 1 / n_trials)
    z2 = norm.ppf(1 - 1 / (n_trials * math.e))
    return sr_std * ((1 - euler) * z1 + euler * z2)


def stationary_bootstrap_ci(
    returns: pd.Series,
    statistic="cagr",
    samples=2000,
    block=20,
    annualization=365,
    seed=42,
) -> tuple[float, float]:
    """Politis-Romano stationary block bootstrap preserving local dependence."""
    x = pd.Series(returns).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    vals = []
    p = 1 / block
    for _ in range(samples):
        idx = np.empty(len(x), dtype=int)
        idx[0] = rng.integers(len(x))
        for j in range(1, len(x)):
            idx[j] = (
                rng.integers(len(x)) if rng.random() < p else (idx[j - 1] + 1) % len(x)
            )
        vals.append(performance(pd.Series(x[idx]), annualization)[statistic])
    return tuple(np.quantile(vals, [0.025, 0.975]))
