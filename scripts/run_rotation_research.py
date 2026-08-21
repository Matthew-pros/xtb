#!/usr/bin/env python
"""Predeclared US-sector rotation experiment. Final holdout may be viewed once."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from alpha_lab.data import download_yahoo_universe
from alpha_lab.ensemble_search import search_ensemble

UNIVERSE = ["SPY", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="1998-12-22")
    p.add_argument(
        "--csv", default=None, help="Optional wide Date + ticker close CSV fallback"
    )
    p.add_argument("--selection-end", default="2014-12-31")
    p.add_argument("--validation-end", default="2019-12-31")
    p.add_argument("--final-end", default=None)
    p.add_argument("--candidates", type=int, default=100_000)
    p.add_argument("--finalists", type=int, default=64)
    p.add_argument("--out", default="artifacts/rotation")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.csv:
        raw = pd.read_csv(a.csv)
        raw["Date"] = pd.to_datetime(raw["Date"])
        px = (
            raw.set_index("Date")
            .reindex(columns=UNIVERSE)
            .apply(pd.to_numeric, errors="raise")
            .sort_index()
        )
        if px.index.has_duplicates:
            raise ValueError("CSV contains duplicate dates")
        px = px.dropna()
        if len(px) < 500 or (px <= 0).any().any():
            raise ValueError(
                "CSV must contain at least 500 common positive rows for every ticker"
            )
    else:
        px = download_yahoo_universe(UNIVERSE, a.start, a.final_end)
    best, audit, metrics, bt = search_ensemble(
        px, a.selection_end, a.validation_end, a.final_end, a.candidates, a.finalists
    )
    audit.to_pickle(out / "selection_audit.pkl")
    bt.to_csv(out / "final_backtest.csv")
    report = {
        "universe": UNIVERSE,
        "data_start": str(px.index.min().date()),
        "data_end": str(px.index.max().date()),
        "selection_end": a.selection_end,
        "validation_end": a.validation_end,
        "best_config": asdict(best),
        "final_holdout": metrics,
        "warning": "Final holdout is consumed. Do not tune after viewing it.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
