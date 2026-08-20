# GARCH drawdown overlay

## Design

A causal rolling Gaussian-QMLE GARCH(1,1) was added to the existing EWMA risk model:

`h[t+1] = omega + alpha * epsilon[t]^2 + beta * h[t]`

- trailing fit window: 730 calendar observations;
- minimum history: 365 observations;
- refit frequency: every 90 observations;
- stationarity enforced by parameter transformation (`alpha + beta < 0.999`);
- forecast at close `t` uses returns only through `t` and sizes return `t+1`;
- failed optimisation falls back to a stationary `(alpha=.06, beta=.92)` model;
- sizing volatility is `max(20d EWMA, GARCH forecast)`.

The final `max` rule is intentional. GARCH is allowed to de-risk the strategy when it sees persistent conditional variance, but is not allowed to manufacture additional leverage merely because its estimate falls below fast EWMA. GARCH forecasts variance, not return direction, and cannot prevent an unanticipated first gap.

## Preliminary BTC comparison

Same unverified public BTC dataset and cost assumptions described in `preliminary_results.md`. Results remain research-only.

### Fixed default parameters, common 2017-03-18–2025-07-24 window

| Metric | EWMA only | EWMA/GARCH conservative max | Change |
|---|---:|---:|---:|
| CAGR | 38.31% | 33.54% | −4.77 pp |
| Volatility | 26.86% | 24.63% | −2.23 pp |
| Sharpe | 1.34 | 1.30 | −0.04 |
| Max drawdown | −31.55% | **−28.77%** | **+2.78 pp** |
| Calmar | 1.21 | 1.17 | −0.05 |

Under identical fixed parameters GARCH reduced drawdown, but paid for it with lower CAGR and slightly lower risk-adjusted performance. It is not free alpha.

### Full nested walk-forward protocol

The complete grid was independently reselected inside each trailing training fold after enabling GARCH.

| Metric | Earlier EWMA OOS | GARCH OOS | Change |
|---|---:|---:|---:|
| CAGR | 21.85% | **23.91%** | +2.06 pp |
| Volatility | 22.87% | **17.50%** | −5.37 pp |
| Sharpe | 0.98 | **1.31** | +0.33 |
| Max drawdown | −28.25% | **−20.00%** | **+8.24 pp** |
| Calmar | 0.77 | **1.20** | +0.42 |
| Worst day | −6.80% | **−5.65%** | +1.15 pp |
| PSR(SR > 0) | 98.86% | **99.90%** | +1.04 pp |
| Bootstrap 95% CAGR | −1.29% to 53.62% | **4.72% to 50.22%** | lower bound > 0 |

OOS period is 2020-03-18–2025-07-24, six folds and 1,953 daily observations. There are 36 disclosed parameter trials per fold. All six GARCH folds selected maximum leverage 1.0×; five selected the lowest 25% target volatility and one selected 50%. This is evidence in favour of conservative sizing, not in favour of leveraging toward a return target.

The GARCH version passed the previously stated positive-bootstrap-lower-bound gate on this preliminary dataset. It still is not institutional certification because the source dataset is not vendor-audited, the sample contains only a few crypto regimes, and no independent venue execution replay was available.

## Interpretation

GARCH helped most in the nested OOS process because conditional volatility persistence reduced exposure for longer after shocks. The result is consistent with the intended use:

- lower exposure during clustered high volatility;
- reduced path dependency entering prolonged drawdowns;
- lower worst-day exposure after volatility has already risen;
- smoother risk budget than a single fast EWMA.

It does **not** predict the first crash and can increase whipsaw/opportunity cost after a rapid V-reversal. Student-t innovations, realised-volatility inputs and asymmetric GJR-GARCH are valid future challenger models, but must be tested as additional disclosed trials and subjected to DSR/PBO correction rather than selected from the same OOS history.
