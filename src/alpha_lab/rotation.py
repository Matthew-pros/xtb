"""Causal multi-asset model library and rotation portfolio engine.

All scores at t use closes through t and weights are shifted one bar before earning
returns. Models emit comparable long-only conviction in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import garch_forecast


def _rsi(px: pd.DataFrame, n: int) -> pd.DataFrame:
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _tema(x: pd.DataFrame, n: int) -> pd.DataFrame:
    a = x.ewm(span=n, adjust=False, min_periods=n).mean()
    b = a.ewm(span=n, adjust=False).mean()
    c = b.ewm(span=n, adjust=False).mean()
    return 3 * a - 3 * b + c


def model_scores(close: pd.DataFrame, benchmark="SPY") -> dict[str, pd.DataFrame]:
    """Predeclared models; no fitted return coefficients and no future data."""
    px = close.astype(float).sort_index()
    ret = px.pct_change()
    mom = lambda n: px.pct_change(n)
    ts = (
        pd.concat(
            [(mom(n) > 0).astype(float) for n in (63, 126, 252)], axis=0, keys=range(3)
        )
        .groupby(level=1)
        .mean()
    )
    r = _rsi(px, 14)
    rsi = ((r - 40) / 30).clip(0, 1) * r.rank(axis=1, pct=True)
    ema = (
        sum(
            (
                px.ewm(span=f, adjust=False).mean()
                > px.ewm(span=s, adjust=False).mean()
            ).astype(float)
            for f, s in ((8, 32), (16, 64), (32, 128), (64, 256))
        )
        / 4
    )
    golden = (px.rolling(50).mean() > px.rolling(200).mean()).astype(float)
    sector = mom(126).rank(axis=1, pct=True) * (mom(252) > 0)
    vol = ret.ewm(span=60, adjust=False, min_periods=60).std() * np.sqrt(252)
    raam = (0.5 * mom(126) + 0.5 * mom(252)).div(vol).rank(axis=1, pct=True) * (
        mom(252) > 0
    )
    sma10 = px > px.rolling(210, min_periods=210).mean()
    ath = px >= 0.95 * px.cummax()
    faber = (sma10.astype(float) + ath.astype(float)) / 2
    carver = (
        sum(
            (
                (
                    px.ewm(span=f, adjust=False).mean()
                    - px.ewm(span=s, adjust=False).mean()
                )
                / (px * ret.ewm(span=32, adjust=False).std())
            ).clip(-2, 2)
            for f, s in ((8, 32), (16, 64), (32, 128), (64, 256))
        )
        / 8
        + 0.5
    )
    carver = carver.clip(0, 1)
    b = ret[benchmark]
    cov = ret.rolling(126).cov(b)
    beta = cov.div(b.rolling(126).var(), axis=0)
    risk_on = (px[benchmark] > px[benchmark].rolling(200).mean()).astype(float)
    beta_rank = beta.rank(axis=1, pct=True)
    beta_rotation = beta_rank.mul(risk_on, axis=0) + (1 - beta_rank).mul(
        1 - risk_on, axis=0
    )
    return {
        "tsmom": ts,
        "rsi": rsi,
        "beta": beta_rotation,
        "ema": ema,
        "golden": golden,
        "sector": sector,
        "raam": raam,
        "faber": faber,
        "carver": carver,
    }


def hmm_risk_probability(
    benchmark_returns: pd.Series, window=1000, min_obs=500, refit=63
) -> pd.Series:
    """Causal probability of a low-volatility/higher-return Gaussian-HMM state.

    Returns and trailing realised volatility are standardised using each refit
    sample. Three deterministic starts reduce local-optimum sensitivity. If every
    fit fails, the last valid model is retained rather than leaking a future refit.
    """
    try:
        import logging

        from hmmlearn.hmm import GaussianHMM
    except ImportError as e:
        raise RuntimeError("Install optional dependency hmmlearn") from e

    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    r = benchmark_returns.astype(float)
    out = pd.Series(np.nan, index=r.index)
    model = None
    state = 0
    scaler_mean = np.zeros(2)
    scaler_std = np.ones(2)
    last = -refit

    def features(x: np.ndarray) -> np.ndarray:
        realised = pd.Series(x).rolling(20, min_periods=20).std().to_numpy()
        return np.c_[x[19:] * 100, realised[19:] * 100]

    for i in range(len(r)):
        x = r.iloc[max(0, i - window + 1) : i + 1].dropna().to_numpy()
        if len(x) < min_obs:
            continue
        z_raw = features(x)
        if model is None or i - last >= refit:
            candidate_mean = z_raw.mean(axis=0)
            candidate_std = z_raw.std(axis=0)
            candidate_std[candidate_std < 1e-8] = 1.0
            z_fit = (z_raw - candidate_mean) / candidate_std
            fits = []
            for seed in (17, 29, 43):
                try:
                    candidate = GaussianHMM(
                        2,
                        covariance_type="diag",
                        min_covar=1e-4,
                        n_iter=300,
                        tol=1e-4,
                        random_state=seed,
                    ).fit(z_fit)
                    score = candidate.score(z_fit)
                    if np.isfinite(score):
                        fits.append((score, candidate))
                except (ValueError, FloatingPointError):
                    continue
            if fits:
                model = max(fits, key=lambda item: item[0])[1]
                scaler_mean, scaler_std = candidate_mean, candidate_std
                raw_state_means = model.means_ * scaler_std + scaler_mean
                # Reward mean return and penalise realised volatility.
                state = int(np.argmax(raw_state_means[:, 0] - raw_state_means[:, 1]))
                last = i
        if model is not None:
            z_now = (z_raw - scaler_mean) / scaler_std
            out.iloc[i] = model.predict_proba(z_now)[-1, state]
    return out.clip(0, 1)


@dataclass(frozen=True)
class RotationConfig:
    model_weights: tuple[float, ...]
    model_names: tuple[str, ...]
    top_k: int = 3
    rebalance_days: int = 5
    score_floor: float = 0.50
    vol_span: int = 60
    target_vol: float = 0.15
    max_asset_weight: float = 0.45
    max_gross: float = 1.0
    fee_bps: float = 10
    slippage_bps: float = 5
    use_garch: bool = True
    use_hmm: bool = True
    hmm_floor: float = 0.25


def run_rotation(
    close: pd.DataFrame,
    cfg: RotationConfig,
    benchmark="SPY",
    precomputed_scores: dict[str, pd.DataFrame] | None = None,
    garch_vol: pd.Series | None = None,
    hmm_prob: pd.Series | None = None,
) -> pd.DataFrame:
    scores = precomputed_scores or model_scores(close, benchmark)
    ret = close.pct_change().fillna(0)
    blend = sum(
        scores[n] * w for n, w in zip(cfg.model_names, cfg.model_weights)
    ) / max(sum(cfg.model_weights), 1e-12)
    vol = ret.ewm(
        span=cfg.vol_span, adjust=False, min_periods=cfg.vol_span
    ).std() * np.sqrt(252)
    g = (
        (
            garch_vol.reindex(ret.index)
            if garch_vol is not None
            else garch_forecast(ret[benchmark], 730, 365, 90, 252)
        )
        if cfg.use_garch
        else pd.Series(0.15, index=ret.index)
    )
    h = (
        (
            hmm_prob.reindex(ret.index)
            if hmm_prob is not None
            else hmm_risk_probability(ret[benchmark])
        )
        if cfg.use_hmm
        else pd.Series(1.0, index=ret.index)
    )
    desired = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    last = np.zeros(close.shape[1])
    for i in range(len(close)):
        if i % cfg.rebalance_days == 0 and np.isfinite(vol.iloc[i]).all():
            s = (
                blend.iloc[i]
                .where(blend.iloc[i] >= cfg.score_floor)
                .dropna()
                .nlargest(cfg.top_k)
            )
            w = pd.Series(0.0, index=close.columns)
            if len(s):
                inv = (s / vol.loc[close.index[i], s.index]).clip(lower=0)
                inv /= inv.sum()
                # Keep residual capital in cash after clipping. Renormalising
                # here would silently violate the per-asset cap.
                w.loc[inv.index] = inv.clip(upper=cfg.max_asset_weight)
                hp = float(h.iloc[i]) if np.isfinite(h.iloc[i]) else 1.0
                hmm_scalar = cfg.hmm_floor + (1 - cfg.hmm_floor) * hp
                gv = float(g.iloc[i])
                if not np.isfinite(gv):
                    gv = float(vol.loc[close.index[i], benchmark])
                risk_scalar = (
                    min(cfg.max_gross, cfg.target_vol / max(gv, 0.05)) * hmm_scalar
                )
                w *= risk_scalar
            last = w.to_numpy()
        desired.iloc[i] = last
    weights = desired.shift(1).fillna(0)  # close t decision earns t+1
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    cost = turnover * (cfg.fee_bps + cfg.slippage_bps) / 10000
    pr = (weights * ret).sum(axis=1) - cost
    result = pd.DataFrame(
        {
            "return": pr,
            "equity": (1 + pr).cumprod(),
            "turnover": turnover,
            "gross": weights.abs().sum(axis=1),
            "hmm_risk_prob": h,
            "garch_vol": g,
        }
    )
    return result.join(weights.add_prefix("weight_"))
