from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import yaml

from .core import ExecutionConfig, StrategyConfig, run_backtest
from .data import download_yahoo, load_csv
from .metrics import performance, probabilistic_sharpe, stationary_bootstrap_ci
from .validation import walk_forward


def _filtered(cls, raw):
    names = {f.name for f in fields(cls)}
    x = {k: v for k, v in raw.items() if k in names}
    if "ema_pairs" in x:
        x["ema_pairs"] = tuple(tuple(p) for p in x["ema_pairs"])
    return cls(**x)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Leakage-safe trend and volatility research"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv")
    src.add_argument("--ticker")
    p.add_argument("--date-col", default="Date")
    p.add_argument("--close-col", default="Close")
    p.add_argument("--config", default="config/strategy.yaml")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--no-walk-forward", action="store_true")
    args = p.parse_args(argv)

    raw = yaml.safe_load(Path(args.config).read_text())
    sc = _filtered(StrategyConfig, raw["strategy"])
    ec = _filtered(ExecutionConfig, raw["execution"])
    vc = raw["validation"]
    close = (
        load_csv(args.csv, args.date_col, args.close_col)
        if args.csv
        else download_yahoo(args.ticker)
    )
    bt = run_backtest(close, sc, ec)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bt.to_csv(out / "full_backtest.csv")
    # Exclude indicator warmup from both strategy and benchmark comparisons.
    first_live = bt["raw_weight"].first_valid_index()
    scored = bt.loc[first_live:]
    report = {
        "data": {
            "start": str(close.index.min().date()),
            "end": str(close.index.max().date()),
            "n": len(close),
            "scored_from": str(first_live.date()),
        },
        "full_sample": performance(scored.strategy_return, ec.periods_per_year),
        "buy_hold": performance(scored.benchmark_return, ec.periods_per_year),
    }
    if not args.no_walk_forward:
        oos, decisions = walk_forward(
            close,
            sc,
            ec,
            vc["grid"],
            vc["train_years"],
            vc["test_years"],
            vc["embargo_days"],
        )
        decisions.to_csv(out / "walk_forward_decisions.csv", index=False)
        oos.to_csv(out / "oos_returns.csv", header=True)
        report["walk_forward_oos"] = performance(oos, ec.periods_per_year)
        report["walk_forward_oos"]["probabilistic_sharpe_gt_0"] = probabilistic_sharpe(
            oos, 0, ec.periods_per_year
        )
        report["walk_forward_oos"]["cagr_bootstrap_95"] = stationary_bootstrap_ci(
            oos,
            samples=vc["bootstrap_samples"],
            annualization=ec.periods_per_year,
            seed=vc["random_seed"],
        )
        report["walk_forward_folds"] = len(decisions)
        report["grid_trials_per_fold"] = (
            len(vc["grid"]["target_vol"])
            * len(vc["grid"]["max_leverage"])
            * len(vc["grid"]["vol_shock_ratio"])
        )
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
