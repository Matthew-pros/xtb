import numpy as np
import pandas as pd

from alpha_lab.core import (
    ExecutionConfig,
    StrategyConfig,
    garch_forecast,
    run_backtest,
    trend_ensemble,
)


def prices(n=900):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    r = 0.0005 + 0.02 * np.sin(np.arange(n) / 30) / 10
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)


def test_weights_are_bounded_and_delayed():
    bt = run_backtest(prices(), StrategyConfig(max_leverage=1.1), ExecutionConfig())
    assert bt.weight.between(0, 1.1).all()
    # First usable desired position cannot earn the same bar's return.
    first = bt.index[bt.desired_weight > 0][0]
    i = bt.index.get_loc(first)
    assert bt.weight.iloc[i] != bt.desired_weight.iloc[i]


def test_costs_reduce_returns():
    p = prices()
    free = run_backtest(
        p, execution=ExecutionConfig(fee_bps=0, slippage_bps=0, annual_funding=0)
    )
    costly = run_backtest(
        p, execution=ExecutionConfig(fee_bps=20, slippage_bps=10, annual_funding=0.1)
    )
    assert costly.equity.iloc[-1] <= free.equity.iloc[-1]


def test_future_mutation_does_not_change_past():
    p = prices()
    a = run_backtest(p)
    p2 = p.copy()
    p2.iloc[-1] *= 5
    b = run_backtest(p2)
    pd.testing.assert_frame_equal(a.iloc[:-1], b.iloc[:-1])


def test_trend_votes_are_long_only():
    f = trend_ensemble(prices(), ((8, 32), (16, 64)))
    assert f.forecast.dropna().between(0, 1).all()


def test_garch_is_positive_and_causal():
    r = prices().pct_change()
    a = garch_forecast(r, window=400, min_obs=200, refit_days=60)
    changed = r.copy()
    changed.iloc[-1] = 0.50
    b = garch_forecast(changed, window=400, min_obs=200, refit_days=60)
    assert (a.dropna() > 0).all()
    pd.testing.assert_series_equal(a.iloc[:-1], b.iloc[:-1])
    assert a.iloc[-1] != b.iloc[-1]
