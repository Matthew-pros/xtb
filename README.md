# Robust Alpha Lab

A reproducible, causal research project with three connected deliverables:

1. a single-asset volatility-managed trend engine;
2. a multi-asset US-sector rotation ensemble with a disclosed 100,000-candidate funnel, GARCH/HMM risk overlays, CSCV/PBO and an untouched final holdout;
3. a Pine Script v6 TEMACD strategy with an optional TEMA challenger and ATR risk overlay.

It is **not** a promise of 90–100% CAGR. No honest process can pre-specify that return and then optimise until history produces it. The Python systems are research/backtest engines; the repository deliberately does not place live orders.

## Single-asset strategy specification

- Daily long/flat trend consensus from EMA pairs 8/32, 16/64, 32/128 and 64/256.
- Consensus forecast is 0, 0.25, 0.50, 0.75 or 1.0. No short crypto leg.
- Carver-style exposure: `forecast × target volatility / ex-ante risk volatility`.
- Risk volatility is conservative `max(20d EWMA, causal rolling GARCH(1,1))`; GARCH may reduce exposure but never increase it.
- Exposure capped by leverage and a six-sigma jump-loss budget.
- Volatility shock scalar, path-dependent drawdown governor and 10% position inertia.
- Signal at close *t* affects close-to-close return *t+1*; never same-bar execution.
- One-way fee + slippage on actual turnover; funding charged only above 1×; idle cash configurable.

The GARCH forecast uses Gaussian QMLE on a trailing two-year window, requires one year of observations, and refits every 90 days using only information known at that close. Defaults target 40% annual volatility and cap gross exposure at 1.25×. These are already aggressive. A 90–100% sustainable CAGR would require an implausibly high Sharpe or ruin-level volatility/leverage.

## Validation protocol

`alpha-lab` writes a full audit trail and performs nested trailing walk-forward selection. Each annual OOS fold selects only `target_vol`, `max_leverage`, and the volatility-shock threshold from the preceding four years, separated by a seven-day embargo. The objective penalises drawdown and turnover; it does not optimise CAGR. Reports include OOS CAGR, Sharpe, Calmar, PSR, and a stationary-block-bootstrap CAGR interval.

This is research-grade and deterministic, but production deployment additionally requires vendor/exchange reconciliation, point-in-time instrument metadata, order-state recovery, broker-specific precision/minimum checks, monitoring, key management, and independent code/risk approval. The repository deliberately does not place live orders.

## Run

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev,rotation]'

# Single-asset trend/GARCH research
.venv/bin/alpha-lab --ticker BTC-USD
# or vetted exchange data:
.venv/bin/alpha-lab --csv data/btc.csv --date-col Date --close-col Close

# Sector ensemble; Yahoo preliminary data
.venv/bin/python scripts/run_rotation_research.py \
  --selection-end 2014-12-31 --validation-end 2019-12-31 \
  --candidates 100000 --finalists 64
# Add --csv sector_etfs.csv when Yahoo is unavailable.

.venv/bin/ruff check src tests scripts
.venv/bin/pytest -q
```

The rotation CSV schema is `Date,SPY,XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY`. See [`docs/rotation_protocol.md`](docs/rotation_protocol.md) for the locked protocol and acceptance gates. The TradingView deliverable is [`tradingview/TQQQ_TEMACD_ATR_v6.pine`](tradingview/TQQQ_TEMACD_ATR_v6.pine).

Single-asset artifacts:

- `full_backtest.csv`: every signal, risk scalar, delayed weight, turnover, cost and return;
- `walk_forward_decisions.csv`: parameters selected in each training fold;
- `oos_returns.csv`: untouched stitched OOS returns;
- `report.json`: performance and statistical validation.

Rotation artifacts are `selection_audit.pkl`, `final_backtest.csv`, and `report.json`; the latter includes bootstrap, doubled-cost, extra-delay and acceptance-gate results.

## Non-negotiable limitations

Free Yahoo data is convenient research data, not an institutional market-data source. A credible crypto cross-sectional test also requires a survivorship-aware historical universe including delistings; selecting today's winners creates enormous bias. For that reason this implementation starts with the liquid BTC winner leg and accepts generic vetted CSV input rather than manufacturing a biased altcoin result.

See [`docs/research.md`](docs/research.md) for evidence, falsification literature, exact rules and deployment gates.
