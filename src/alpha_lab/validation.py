"""Nested walk-forward selection. Test folds are never used to choose parameters."""

from __future__ import annotations

from itertools import product

import pandas as pd

from .core import ExecutionConfig, StrategyConfig, run_backtest, with_params
from .metrics import performance


def parameter_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]


def robust_score(ret: pd.Series, turnover: pd.Series, annualization=365) -> float:
    m = performance(ret, annualization)
    # Penalise tail risk and implementation intensity; do not optimise CAGR directly.
    return float(
        m["sharpe"]
        - 0.50 * abs(m["max_drawdown"])
        - 0.02 * turnover.mean() * annualization
    )


def walk_forward(
    close: pd.Series,
    base: StrategyConfig,
    execution: ExecutionConfig,
    grid: dict[str, list],
    train_years=4,
    test_years=1,
    embargo_days=7,
) -> tuple[pd.Series, pd.DataFrame]:
    """Anchored calendar walk-forward with an embargo between train and test.

    Parameters are selected on trailing training windows. The returned OOS series is a
    concatenation of untouched test years. Warmup data precedes each test calculation,
    but scoring starts strictly after the embargo.
    """
    close = close.sort_index()
    start = close.index.min() + pd.DateOffset(years=train_years)
    end = close.index.max()
    oos, decisions = [], []
    test_start = start
    candidates = parameter_grid(grid)
    while test_start < end:
        train_start = test_start - pd.DateOffset(years=train_years)
        train_end = test_start - pd.Timedelta(days=embargo_days)
        test_end = min(
            test_start + pd.DateOffset(years=test_years), end + pd.Timedelta(days=1)
        )
        train_px = close.loc[train_start:train_end]
        if len(train_px) < 500:
            test_start = test_end
            continue
        scored = []
        for params in candidates:
            cfg = with_params(base, **params)
            bt = run_backtest(train_px, cfg, execution)
            scored.append(
                (
                    robust_score(
                        bt.strategy_return, bt.turnover, execution.periods_per_year
                    ),
                    params,
                )
            )
        score, best = max(scored, key=lambda z: z[0])
        cfg = with_params(base, **best)
        # Include prior history only for indicators; retain only untouched test observations.
        history = (
            close.loc[:test_end].iloc[:-1] if test_end > end else close.loc[:test_end]
        )
        bt = run_backtest(history, cfg, execution)
        fold = bt.loc[
            (bt.index >= test_start) & (bt.index < test_end), "strategy_return"
        ]
        if len(fold):
            oos.append(fold)
            decisions.append(
                {
                    "test_start": test_start,
                    "test_end": test_end,
                    "train_score": score,
                    **best,
                    "oos_n": len(fold),
                }
            )
        test_start = test_end
    if not oos:
        raise ValueError("No walk-forward folds; provide a longer history")
    return pd.concat(oos).sort_index(), pd.DataFrame(decisions)
