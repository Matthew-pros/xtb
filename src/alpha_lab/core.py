"""Causal signal, risk sizing and execution-aware backtest.

Every position used for return t is computed only with information through t-1.
The implementation intentionally uses a long/flat winner leg: shorting crypto
losers has weak evidence and introduces liquidation/borrow risks.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class StrategyConfig:
    ema_pairs: tuple[tuple[int, int], ...] = ((8, 32), (16, 64), (32, 128), (64, 256))
    target_vol: float = 0.40
    vol_span_fast: int = 20
    vol_span_slow: int = 120
    garch_enabled: bool = True
    garch_window: int = 730
    garch_min_obs: int = 365
    garch_refit_days: int = 90
    garch_vol_multiplier: float = 1.0
    max_leverage: float = 1.25
    max_jump_loss: float = 0.20
    jump_sigma: float = 6.0
    vol_shock_ratio: float = 1.75
    vol_shock_multiplier: float = 0.50
    drawdown_soft: float = 0.15
    drawdown_hard: float = 0.30
    drawdown_floor: float = 0.25
    trade_buffer: float = 0.10


@dataclass(frozen=True)
class ExecutionConfig:
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    annual_funding: float = 0.06
    annual_cash_rate: float = 0.0
    periods_per_year: int = 365


def validate_prices(close: pd.Series) -> pd.Series:
    x = pd.Series(close, dtype=float).sort_index()
    if x.index.has_duplicates:
        raise ValueError("Duplicate timestamps")
    if x.isna().any() or (x <= 0).any():
        raise ValueError("Prices must be finite, positive and complete")
    if len(x) < 300:
        raise ValueError("At least 300 observations are required")
    return x


def trend_ensemble(close: pd.Series, pairs: Iterable[tuple[int, int]]) -> pd.DataFrame:
    """Continuous long-only consensus in [0, 1], known at each close."""
    votes = {}
    for fast, slow in pairs:
        if not 1 < fast < slow:
            raise ValueError(f"Invalid EMA pair {(fast, slow)}")
        f = close.ewm(span=fast, adjust=False, min_periods=slow).mean()
        s = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
        votes[f"ema_{fast}_{slow}"] = (f > s).astype(float).where(s.notna())
    frame = pd.DataFrame(votes)
    frame["forecast"] = frame.mean(axis=1, skipna=False)
    return frame


def _garch_transform(theta: np.ndarray) -> tuple[float, float, float]:
    """Map unconstrained parameters to a stationary GARCH(1,1)."""
    omega = float(np.exp(np.clip(theta[0], -25, 10)))
    alpha = float(0.35 / (1.0 + np.exp(-np.clip(theta[1], -30, 30))))
    # Guarantees alpha + beta < 0.999 without a fragile optimiser constraint.
    beta = float((0.999 - alpha) / (1.0 + np.exp(-np.clip(theta[2], -30, 30))))
    return omega, alpha, beta


def _garch_filter(x: np.ndarray, params: tuple[float, float, float]) -> np.ndarray:
    omega, alpha, beta = params
    h = np.empty(len(x), dtype=float)
    unconditional = omega / max(1.0 - alpha - beta, 1e-6)
    h[0] = max(unconditional, np.var(x), 1e-8)
    for j in range(1, len(x)):
        h[j] = max(omega + alpha * x[j - 1] ** 2 + beta * h[j - 1], 1e-10)
    return h


def _fit_garch(x: np.ndarray) -> tuple[float, float, float]:
    """Gaussian QMLE. Input is demeaned percentage returns for numerical stability."""
    variance = max(float(np.var(x)), 1e-6)

    def objective(theta: np.ndarray) -> float:
        params = _garch_transform(theta)
        h = _garch_filter(x, params)
        return float(0.5 * np.sum(np.log(h) + x * x / h))

    # Starts around alpha=.06, beta=.92 and long-run sample variance.
    alpha0, beta0 = 0.06, 0.92
    omega0 = variance * (1 - alpha0 - beta0)
    t0 = np.array(
        [
            np.log(max(omega0, 1e-10)),
            np.log(alpha0 / (0.35 - alpha0)),
            np.log(beta0 / (0.999 - alpha0 - beta0)),
        ]
    )
    result = minimize(
        objective, t0, method="L-BFGS-B", options={"maxiter": 250, "ftol": 1e-9}
    )
    params = _garch_transform(result.x)
    if not result.success or not np.isfinite(result.fun) or sum(params[1:]) >= 0.999:
        # A deterministic stationary fallback is safer than dropping risk control.
        return variance * 0.02, 0.06, 0.92
    return params


def garch_forecast(
    returns: pd.Series,
    window: int = 730,
    min_obs: int = 365,
    refit_days: int = 90,
    annualization: int = 365,
) -> pd.Series:
    """Causal one-step GARCH(1,1) forecast with rolling, scheduled QMLE refits.

    Forecast at index t uses returns through t and is intended to size the t+1 position.
    Parameters are fitted only on a trailing window, preventing future regime leakage.
    """
    if min_obs < 100 or window < min_obs or refit_days < 1:
        raise ValueError("Require window >= min_obs >= 100 and refit_days >= 1")
    r = pd.Series(returns, dtype=float)
    out = pd.Series(np.nan, index=r.index, name="garch_vol")
    params = None
    last_fit = -refit_days
    h_next = np.nan
    mean = 0.0
    for i in range(len(r)):
        if not np.isfinite(r.iloc[i]):
            continue
        start = max(0, i - window + 1)
        sample = r.iloc[start : i + 1].dropna().to_numpy() * 100.0
        if len(sample) < min_obs:
            continue
        if params is None or i - last_fit >= refit_days:
            mean = float(sample.mean())
            x = sample - mean
            params = _fit_garch(x)
            h = _garch_filter(x, params)
            h_next = params[0] + params[1] * x[-1] ** 2 + params[2] * h[-1]
            last_fit = i
        else:
            innovation = r.iloc[i] * 100.0 - mean
            h_next = params[0] + params[1] * innovation**2 + params[2] * h_next
        out.iloc[i] = np.sqrt(max(h_next, 0.0)) / 100.0 * np.sqrt(annualization)
    return out


def _risk_inputs(
    close: pd.Series, cfg: StrategyConfig, annualization: int
) -> pd.DataFrame:
    r = close.pct_change()
    vf = r.ewm(span=cfg.vol_span_fast, adjust=False, min_periods=cfg.vol_span_fast).std(
        bias=False
    ) * np.sqrt(annualization)
    vs = r.ewm(span=cfg.vol_span_slow, adjust=False, min_periods=cfg.vol_span_slow).std(
        bias=False
    ) * np.sqrt(annualization)
    if cfg.garch_enabled:
        vg = (
            garch_forecast(
                r,
                cfg.garch_window,
                cfg.garch_min_obs,
                cfg.garch_refit_days,
                annualization,
            )
            * cfg.garch_vol_multiplier
        )
        # Conservative ensemble: never size from a volatility estimate below fast EWMA.
        risk_vol = pd.concat([vf.rename("ewma"), vg], axis=1).max(axis=1, skipna=False)
    else:
        vg = pd.Series(np.nan, index=close.index, name="garch_vol")
        risk_vol = vf
    signal = trend_ensemble(close, cfg.ema_pairs)["forecast"]
    # Carver-style forecast / volatility. A jump cap prevents absurd leverage in calm regimes.
    vol_weight = cfg.target_vol / risk_vol.replace(0, np.nan)
    jump_weight = cfg.max_jump_loss / (
        (risk_vol / np.sqrt(annualization)) * cfg.jump_sigma
    ).replace(0, np.nan)
    shock = pd.Series(
        np.where(risk_vol > cfg.vol_shock_ratio * vs, cfg.vol_shock_multiplier, 1.0),
        index=close.index,
    )
    raw = (signal * np.minimum(vol_weight, jump_weight) * shock).clip(
        0, cfg.max_leverage
    )
    return pd.DataFrame(
        {
            "asset_return": r,
            "forecast": signal,
            "fast_vol": vf,
            "slow_vol": vs,
            "garch_vol": vg,
            "risk_vol": risk_vol,
            "vol_shock_scalar": shock,
            "raw_weight": raw,
        }
    )


def run_backtest(
    close: pd.Series,
    strategy: StrategyConfig | None = None,
    execution: ExecutionConfig | None = None,
) -> pd.DataFrame:
    """Close-to-close simulation with one-bar delayed weights and explicit turnover costs.

    The equity drawdown governor is path-dependent and therefore evaluated sequentially.
    A 10% no-trade buffer is applied against the current exposure. Leverage funding only
    applies above 100%; idle cash earns the configured cash rate.
    """
    strategy = strategy or StrategyConfig()
    execution = execution or ExecutionConfig()
    close = validate_prices(close)
    f = _risk_inputs(close, strategy, execution.periods_per_year)
    out = f.copy()
    n = len(out)
    weight = np.zeros(n)
    desired = np.zeros(n)
    dd_scalar = np.ones(n)
    turnover = np.zeros(n)
    costs = np.zeros(n)
    net = np.zeros(n)
    equity = np.ones(n)
    peak = 1.0
    one_way_cost = (execution.fee_bps + execution.slippage_bps) / 10_000
    cash_daily = execution.annual_cash_rate / execution.periods_per_year
    funding_daily = execution.annual_funding / execution.periods_per_year

    raw = out["raw_weight"].to_numpy()
    asset_r = out["asset_return"].fillna(0).to_numpy()
    current = 0.0
    for i in range(1, n):
        # Position carried during t was decided after close t-1.
        weight[i] = current
        turnover[i] = abs(weight[i] - weight[i - 1])
        costs[i] = (
            turnover[i] * one_way_cost + max(weight[i] - 1.0, 0.0) * funding_daily
        )
        net[i] = (
            weight[i] * asset_r[i] + max(1.0 - weight[i], 0.0) * cash_daily - costs[i]
        )
        # Bankruptcy is represented rather than silently producing invalid compounding.
        equity[i] = max(equity[i - 1] * (1.0 + net[i]), 0.0)
        peak = max(peak, equity[i])
        dd = 1.0 - equity[i] / peak if peak else 1.0
        if dd <= strategy.drawdown_soft:
            scalar = 1.0
        elif dd >= strategy.drawdown_hard:
            scalar = strategy.drawdown_floor
        else:
            frac = (dd - strategy.drawdown_soft) / (
                strategy.drawdown_hard - strategy.drawdown_soft
            )
            scalar = 1.0 - frac * (1.0 - strategy.drawdown_floor)
        dd_scalar[i] = scalar
        target = (raw[i] if np.isfinite(raw[i]) else 0.0) * scalar
        desired[i] = target
        # Carver position inertia: ignore changes smaller than 10% of current risk,
        # with a small absolute floor so a flat book can enter.
        buffer = max(abs(current) * strategy.trade_buffer, 0.01)
        if abs(target - current) > buffer:
            current = target

    out["desired_weight"] = desired
    out["weight"] = weight
    out["drawdown_scalar"] = dd_scalar
    out["turnover"] = turnover
    out["cost"] = costs
    out["strategy_return"] = net
    out["equity"] = equity
    out["benchmark_return"] = out["asset_return"].fillna(0)
    out.attrs["strategy_config"] = strategy
    out.attrs["execution_config"] = execution
    return out


def with_params(cfg: StrategyConfig, **kwargs) -> StrategyConfig:
    return replace(cfg, **kwargs)
