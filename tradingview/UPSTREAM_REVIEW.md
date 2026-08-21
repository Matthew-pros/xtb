# Review of Niemiec–Michalowski Capital Management/TEMACD

Reviewed upstream commit: `b06c4d2d0caca5d0a05ec0dc48b0e2295a1b2e68` (the repository contained one commit at review time).

## Useful ideas incorporated

- Component signals are deliberately simple and interpretable.
- MACD and three-speed moving-average events are combined with OR, matching the supplied Pine concept.
- Signals are shifted one row before the vectorbt simulation, which is directionally correct protection against same-bar use.
- Indicator warmup is downloaded before the scored interval.
- Fees and slippage are disclosed as 0.1% each in YAML.
- IS/OOS split, parameter sensitivity plots, underwater charts, Ulcer and Calmar metrics are useful reporting components.
- Upstream uses TA-Lib **TEMA**, not EMA. An optional TEMA engine was therefore added to Pine without changing the default EMA baseline.

## Material differences from the supplied Pine strategy

| Topic | Supplied Pine | Linked repository |
|---|---|---|
| MA | EMA | TA-Lib TEMA |
| Directions | configurable long/short/both | long/flat |
| Fill model | next-bar TradingView order model | shifted boolean on daily closes |
| Costs | absent | 10 bp fee + 10 bp slippage |
| Optimisation | fixed parameters | very large exhaustive grids |
| Ensemble | all event types OR'ed | TEMA and MACD event streams OR'ed |

These are not interchangeable replications. TEMA has much less lag and more overshoot than EMA, so it must be reported as another tested model.

## Validation concerns

1. The configured MACD grid contains roughly 98,000 combinations and TEMA roughly 460,000 raw combinations before validity filtering. Selecting maximum in-sample Sharpe from that many trials creates severe multiple-testing bias. One chronological holdout does not erase that bias.
2. The README's instruction that an upward line after one split proves robustness is too weak. Robustness requires repeated walk-forward/CPCV folds, neighbourhood stability, cost/delay stress and DSR/PBO correction.
3. `optimization/validator.py` explicitly returns the parameters with the best OOS Sharpe among the top IS candidates. If used for a final reported result, that OOS period becomes validation data, not untouched test data.
4. `sensitivity.py` repeatedly examines OOS performance around selected parameters. Once viewed and used to make choices, that period is no longer pristine holdout data.
5. TEMA validation enforces `w1 < w2` but does not enforce `w2 < w3`, despite naming the third speed “slow”.
6. `vectorbt.from_signals` starts each standalone OOS portfolio flat. That can differ from the position a continuously running live strategy carries across the split.
7. A one-row signal shift with close-price simulation is conservative in information timing but is not identical to Pine's next-open fill. Gap returns and percent slippage need an explicit parity test.
8. The repository has a short history (one commit), no automated tests visible, and suppresses broad exceptions in several places. It is useful research inspiration, not independent verification of alpha.

## Recommended fair comparison

Run four predeclared models, each with identical costs and next-open timing:

1. original EMA + MACD, no ATR;
2. original EMA + MACD, ATR overlay;
3. TEMA + MACD, no ATR;
4. TEMA + MACD, ATR overlay.

Freeze a small coarse parameter grid on each training fold. Pool results across folds and report every trial count. The incremental ATR claim is valid only if model 2 improves model 1 and model 4 improves model 3 across several OOS folds—not merely on full-sample TQQQ CAGR.
