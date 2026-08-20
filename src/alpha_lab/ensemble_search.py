"""100k-candidate funnel with a single untouched final holdout.

The large search is a disclosed multiple-testing exercise, not proof of alpha. A cheap
linear sleeve proxy screens candidates; only finalists receive exact path-dependent
simulation. Selection ends before the final holdout starts.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from itertools import combinations

import numpy as np
import pandas as pd

from .metrics import (
    expected_max_sharpe,
    performance,
    probabilistic_sharpe,
    stationary_bootstrap_ci,
)
from .rotation import (
    RotationConfig,
    garch_forecast,
    hmm_risk_probability,
    model_scores,
    run_rotation,
)

MODEL_NAMES = (
    "tsmom",
    "rsi",
    "beta",
    "ema",
    "golden",
    "sector",
    "raam",
    "faber",
    "carver",
)


def cscv_pbo(candidate_returns: pd.DataFrame, blocks=8) -> float:
    """Bailey et al. CSCV estimate on exact finalists; lower is better."""
    x = candidate_returns.dropna(how="all").fillna(0)
    parts = np.array_split(np.arange(len(x)), blocks)
    logits = []
    for train_blocks in combinations(range(blocks), blocks // 2):
        test_blocks = set(range(blocks)) - set(train_blocks)
        ii = np.concatenate([parts[i] for i in train_blocks])
        oo = np.concatenate([parts[i] for i in test_blocks])

        def sr(a):
            sd = a.std(axis=0, ddof=1)
            return np.divide(
                a.mean(axis=0), sd, out=np.full(a.shape[1], -np.inf), where=sd > 0
            )

        winner = int(np.argmax(sr(x.to_numpy()[ii])))
        oos = sr(x.to_numpy()[oo])
        rank = (np.sum(oos <= oos[winner]) - 0.5) / len(oos)
        rank = np.clip(rank, 1e-6, 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    return float(np.mean(np.asarray(logits) < 0))


def candidate_grid(n=100_000, seed=20260819) -> list[RotationConfig]:
    """Deterministic mixed discrete grid; exactly n disclosed configurations."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        # Sparse integer weights prevent meaningless 9-model equal soup.
        w = rng.integers(0, 4, len(MODEL_NAMES))
        if not w.any():
            w[rng.integers(len(w))] = 1
        out.append(
            RotationConfig(
                tuple(map(float, w)),
                MODEL_NAMES,
                top_k=int(rng.choice([2, 3, 4])),
                rebalance_days=int(rng.choice([5, 10, 21])),
                score_floor=float(rng.choice([0.45, 0.50, 0.55, 0.60])),
                vol_span=int(rng.choice([20, 40, 60, 90])),
                target_vol=float(rng.choice([0.10, 0.12, 0.15, 0.18])),
                max_asset_weight=float(rng.choice([0.35, 0.45, 0.60])),
                max_gross=float(rng.choice([0.75, 1.0])),
                use_garch=True,
                use_hmm=True,
            )
        )
    return out


def _sleeve_returns(
    close: pd.DataFrame, scores: dict[str, pd.DataFrame], names=MODEL_NAMES
) -> pd.DataFrame:
    ret = close.pct_change().fillna(0)
    vol = ret.ewm(span=60, adjust=False, min_periods=60).std()
    ans = {}
    for name in names:
        s = scores[name]
        w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        for i in range(0, len(close), 5):
            # Set the held block [i, next rebalance) explicitly; never leak a
            # previously selected asset into later blocks.
            j = min(i + 5, len(close))
            w.iloc[i:j, :] = 0
            pick = s.iloc[i].where(s.iloc[i] >= 0.5).dropna().nlargest(3)
            if len(pick) and np.isfinite(vol.iloc[i][pick.index]).all():
                z = pick / vol.iloc[i][pick.index]
                z /= z.sum()
                w.iloc[i:j, w.columns.get_indexer(z.index)] = z.values
        ans[name] = (w.shift(1).fillna(0) * ret).sum(axis=1)
    return pd.DataFrame(ans)


def search_ensemble(
    close: pd.DataFrame,
    selection_end: str,
    validation_end: str,
    final_end: str | None = None,
    n_candidates=100_000,
    finalists=64,
    benchmark="SPY",
):
    """Search on pre-selection history, rank exact finalists on validation, test once.

    Returns the selected config, audit table and untouched final-holdout metrics.
    Caller must not iterate after viewing final metrics.
    """
    if not close.index.is_monotonic_increasing:
        close = close.sort_index()
    sel = pd.Timestamp(selection_end)
    val = pd.Timestamp(validation_end)
    if not (close.index.min() < sel < val < close.index.max()):
        raise ValueError("Need ordered selection, validation and final periods")
    scores = model_scores(close, benchmark)
    sleeves = _sleeve_returns(close, scores)
    train = sleeves.loc[:sel].dropna()
    candidates = candidate_grid(n_candidates)
    # BLAS-vectorised chunks avoid 100k Python backtests while bounding memory.
    cuts = np.array_split(np.arange(len(train)), 3)
    ranked = []
    x = train.to_numpy()
    chunk = 2000
    for start in range(0, n_candidates, chunk):
        batch = candidates[start : start + chunk]
        w = np.asarray([c.model_weights for c in batch], dtype=float)
        w /= w.sum(axis=1, keepdims=True)
        block_sr = []
        for ix in cuts:
            y = x[ix] @ w.T
            sd = y.std(axis=0, ddof=1)
            block_sr.append(
                np.divide(
                    y.mean(axis=0), sd, out=np.full(len(batch), -99.0), where=sd > 0
                )
                * np.sqrt(252)
            )
        score = (
            np.min(np.vstack(block_sr), axis=0) - np.count_nonzero(w, axis=1) * 0.015
        )
        ranked.extend((float(v), start + k) for k, v in enumerate(score))
    keep = sorted(ranked, reverse=True)[:finalists]
    g = garch_forecast(close[benchmark].pct_change(), 730, 365, 90, 252)
    h = hmm_risk_probability(close[benchmark].pct_change())
    audit = []
    exact_returns = {}
    for proxy, j in keep:
        c = candidates[j]
        bt = run_rotation(
            close.loc[:val],
            c,
            benchmark,
            {k: v.loc[:val] for k, v in scores.items()},
            g.loc[:val],
            h.loc[:val],
        )
        rv = bt.loc[(bt.index > sel) & (bt.index <= val), "return"]
        exact_returns[j] = bt.loc[bt.index <= val, "return"]
        m = performance(rv, 252)
        validation_turnover = bt.loc[
            (bt.index > sel) & (bt.index <= val), "turnover"
        ].mean()
        score = (
            m["sharpe"]
            - 0.75 * abs(m["max_drawdown"])
            - 0.02 * validation_turnover * 252
        )
        audit.append(
            {
                "candidate": j,
                "proxy_score": proxy,
                "validation_score": score,
                **m,
                "config": asdict(c),
            }
        )
    tab = pd.DataFrame(audit).sort_values("validation_score", ascending=False)
    best = candidates[int(tab.iloc[0].candidate)]
    end = pd.Timestamp(final_end) if final_end else close.index.max()
    bt = run_rotation(close.loc[:end], best, benchmark, scores, g, h)
    final = bt.loc[(bt.index > val) & (bt.index <= end), "return"]
    metrics = performance(final, 252)
    metrics["psr_gt_zero"] = probabilistic_sharpe(final, 0, 252)
    metrics["trials_disclosed"] = n_candidates
    metrics["finalists_cscv_pbo"] = cscv_pbo(pd.DataFrame(exact_returns).loc[:val])
    metrics["expected_max_sr_null_approx"] = expected_max_sharpe(
        n_candidates, 1 / np.sqrt(max(len(train) / 252, 1))
    )
    metrics["cagr_bootstrap_95"] = stationary_bootstrap_ci(
        final, samples=1000, block=20, annualization=252, seed=20260819
    )

    doubled = replace(
        best, fee_bps=best.fee_bps * 2, slippage_bps=best.slippage_bps * 2
    )
    doubled_bt = run_rotation(close.loc[:end], doubled, benchmark, scores, g, h)
    doubled_final = doubled_bt.loc[
        (doubled_bt.index > val) & (doubled_bt.index <= end), "return"
    ]
    metrics["doubled_cost_cagr"] = performance(doubled_final, 252)["cagr"]

    final_bt = bt.loc[(bt.index > val) & (bt.index <= end)]
    delayed_weights = final_bt.filter(like="weight_").shift(1).fillna(0)
    delayed_turnover = delayed_weights.diff().abs().sum(axis=1).fillna(0)
    delayed_asset_returns = close.loc[final_bt.index].pct_change().fillna(0)
    delayed_asset_returns.columns = [f"weight_{c}" for c in delayed_asset_returns]
    delayed_return = (delayed_weights * delayed_asset_returns).sum(axis=1) - (
        delayed_turnover * (best.fee_bps + best.slippage_bps) / 10_000
    )
    metrics["extra_bar_delay_cagr"] = performance(delayed_return, 252)["cagr"]

    lower_cagr = metrics["cagr_bootstrap_95"][0]
    metrics["acceptance_passed"] = bool(
        metrics["max_drawdown"] >= -0.25
        and metrics["sharpe"] >= 1.0
        and metrics["calmar"] >= 0.7
        and metrics["finalists_cscv_pbo"] < 0.10
        and lower_cagr > 0
        and metrics["doubled_cost_cagr"] > 0
        and metrics["extra_bar_delay_cagr"] > 0
    )
    return best, tab, metrics, bt
