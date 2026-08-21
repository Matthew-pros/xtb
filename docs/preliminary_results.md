# Preliminary BTC run (not an institutional certification)

> This document is the pre-GARCH baseline. See [`garch_results.md`](garch_results.md) for the new conservative EWMA/GARCH overlay and updated walk-forward results.

Run date: 2026-08-20. This smoke test used 3,415 daily BTC closes from 2016-03-18 through 2025-07-24 assembled from two public GitHub/Yahoo-derived files because Yahoo and exchange APIs were unavailable from the sandbox. The overlap was continuous, but provenance, corrections, venue, timestamp and point-in-time availability were not independently certified. The data are therefore **not committed and these figures must not be represented as audited performance**.

Assumptions: next-bar close-to-close execution, 10 bp fee + 5 bp slippage per one-way turnover, 6% annual funding above 1×, 0% cash return. Full-sample scoring begins after the 256-day indicator warmup.

| Metric | Strategy full sample | BTC buy & hold | Stitched walk-forward OOS |
|---|---:|---:|---:|
| Period | 2016-11-28–2025-07-24 | same | 2020-03-18–2025-07-24 |
| CAGR | 44.95% | 79.92% | **21.85%** |
| Annual volatility | 27.74% | 70.02% | **22.87%** |
| Sharpe | 1.48 | 1.19 | **0.98** |
| Maximum drawdown | −31.55% | −83.40% | **−28.25%** |
| Calmar | 1.42 | 0.96 | **0.77** |
| Worst day | −9.21% | −37.17% | **−6.80%** |
| Total return | 2,388% | 16,054% | **188%** |

OOS PSR that Sharpe is above zero was 98.86%. The stationary-block-bootstrap 95% interval for OOS CAGR was **−1.29% to +53.62%**. Because that interval includes zero, this preliminary run fails the memo's strict bootstrap acceptance gate despite attractive point estimates. It does not validate a 90–100% CAGR claim.

## Nested grid decisions

Each fold used only its trailing four-year training window, then held parameters fixed for the next year. There were 36 disclosed trials per fold; EMA speeds were fixed rather than mined.

| OOS fold starts | Target vol | Max leverage | Vol-shock threshold |
|---|---:|---:|---:|
| 2020-03-18 | 25% | 1.50× | 1.50× |
| 2021-03-18 | 25% | 1.25× | 1.75× |
| 2022-03-18 | 25% | 1.25× | 1.75× |
| 2023-03-18 | 25% | 1.25× | 1.50× |
| 2024-03-18 | 50% | 1.50× | 1.75× |
| 2025-03-18 | 25% | 1.00× | 1.75× |

The unstable 2024 choice is another warning against declaring the grid optimum structural. Most folds preferred the lowest risk target. Raising leverage merely to force 90–100% CAGR would contradict the evidence and the stated risk objective.
