import numpy as np
import pandas as pd

from alpha_lab.ensemble_search import (
    MODEL_NAMES,
    candidate_grid,
    cscv_pbo,
    search_ensemble,
)
from alpha_lab.rotation import RotationConfig, model_scores, run_rotation


def panel(n=900):
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    r = rng.normal(0.0003, 0.012, (n, 5))
    return pd.DataFrame(
        100 * np.exp(np.cumsum(r, axis=0)),
        index=idx,
        columns=["SPY", "XLK", "XLF", "XLE", "XLV"],
    )


def test_all_predeclared_models_are_causal_and_bounded():
    p = panel()
    a = model_scores(p)
    p2 = p.copy()
    p2.iloc[-1] *= 2
    b = model_scores(p2)
    assert set(a) == set(MODEL_NAMES)
    for k in a:
        assert a[k].stack().dropna().between(0, 1).all()
        pd.testing.assert_frame_equal(a[k].iloc[:-1], b[k].iloc[:-1])


def test_rotation_has_delayed_bounded_exposure():
    p = panel()
    s = model_scores(p)
    c = RotationConfig(
        (1.0,) * 9, MODEL_NAMES, use_garch=False, use_hmm=False, max_gross=1.0
    )
    b = run_rotation(p, c, precomputed_scores=s)
    assert not b["return"].isna().any()
    assert b.gross.max() <= 1.000001
    assert b.filter(like="weight_").max().max() <= c.max_asset_weight + 1e-9


def test_candidate_count_is_disclosed_exactly():
    c = candidate_grid(1000)
    assert len(c) == 1000 and all(sum(x.model_weights) > 0 for x in c)


def test_cscv_pbo_is_probability():
    rng = np.random.default_rng(9)
    r = pd.DataFrame(rng.normal(0, 0.01, (800, 12)))
    p = cscv_pbo(r)
    assert 0 <= p <= 1


def test_small_search_runs_end_to_end():
    p = panel(1400)
    # Add enough assets for top-k candidates while retaining SPY benchmark.
    for j, name in enumerate(["XLB", "XLI", "XLP", "XLU", "XLY"]):
        p[name] = p["SPY"] * (1 + 0.0001 * (j + 1)) ** np.arange(len(p))
    best, audit, metrics, backtest = search_ensemble(
        p,
        str(p.index[700].date()),
        str(p.index[1050].date()),
        n_candidates=20,
        finalists=3,
    )
    assert len(audit) == 3
    assert metrics["observations"] > 100
    assert isinstance(metrics["acceptance_passed"], bool)
    assert not backtest["return"].isna().any()
    assert backtest.filter(like="weight_").max().max() <= best.max_asset_weight + 1e-9
