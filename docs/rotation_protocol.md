# Multi-asset sector rotation — locked research protocol

Mandate selected by user: liquid US sector ETF rotation, preliminary Yahoo adjusted-close data, maximum OOS drawdown target 25%, selection by risk-adjusted robustness rather than CAGR.

## Universe

`SPY, XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY`. Common complete history is used; an ETF is never silently treated as cash before inception. This excludes XLRE to retain a materially longer common sample. A production version needs a total-return vendor and point-in-time constituent/instrument records.

## Predeclared sleeves

- time-series momentum: mean of positive 3/6/12-month own-return votes;
- RSI rotation: RSI(14) strength combined with cross-sectional percentile;
- beta rotation: high beta in SPY risk-on regime, low beta in risk-off regime;
- EMA ensemble: 8/32, 16/64, 32/128 and 64/256 votes;
- golden cross: SMA 50 above SMA 200;
- sector rotation: six-month relative momentum subject to positive 12-month momentum;
- RAAM: cross-sectional rank of 6/12-month momentum divided by EWMA volatility, with absolute-momentum filter;
- Faber: equal blend of 10-month SMA and within-5%-of-ATH rules;
- Carver: continuous, capped multi-speed EWMAC forecast;
- GARCH: causal rolling GARCH(1,1) portfolio-volatility scaler;
- HMM: causal rolling two-state Gaussian regime probability with 25% exposure floor.

Every directional score is bounded to [0,1]. At rebalance the highest-scoring 2–4 sectors above a floor are inverse-volatility weighted and capped per asset. GARCH and HMM affect portfolio gross risk, not return direction. Weights are shifted one bar and pay 10 bp fee plus 5 bp slippage per one-way turnover.

## 100,000-combination funnel

One hundred thousand deterministic candidate configurations are generated with a disclosed seed. The search varies sparse model weights, top-K, rebalance frequency, score floor, volatility window, target volatility, asset cap and gross cap. It does not fit arbitrary return coefficients.

To make the search computationally and statistically tractable:

1. a cheap predeclared model-sleeve proxy ranks all 100,000 candidates on the selection period using the worst Sharpe across three chronological blocks and a model-complexity penalty;
2. only 64 finalists receive the exact path-dependent GARCH/HMM/turnover simulation;
3. one validation period selects the finalist using Sharpe minus drawdown and turnover penalties;
4. CSCV/PBO is reported across exact finalists;
5. the final holdout is evaluated exactly once and must not be used for another iteration;
6. the report discloses all 100,000 trials and an expected-maximum-Sharpe null approximation.

This reduces but cannot abolish data mining. A search of 100,000 variants is intrinsically a high multiple-testing exercise. Passing requirements are final-holdout MaxDD no worse than −25%, Sharpe at least 1, Calmar at least 0.7, PBO below 10%, positive dependent-bootstrap lower CAGR bound, and survival under doubled costs and one-bar extra delay.

## Run

```bash
python -m venv .venv
.venv/bin/pip install -e '.[rotation]'
.venv/bin/python scripts/run_rotation_research.py \
  --selection-end 2014-12-31 \
  --validation-end 2019-12-31 \
  --candidates 100000 --finalists 64
```

The current sandbox could not establish TLS to Yahoo, so no performance result has been fabricated. Run artifacts are created only when the selected data source responds. The final ensemble must not be ported to Pine until this protocol completes and acceptance gates pass.
