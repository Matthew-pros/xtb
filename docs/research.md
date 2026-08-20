# Research decision memo

## Conclusion first

The selected prior is **diversified time-series trend**, applied long/flat to BTC, with volatility-managed sizing. It has stronger cross-market and out-of-sample support than a bespoke indicator stack. BTC is selected as the first implementation asset because (a) it has positive long-run drift and deep liquidity, (b) crypto trend evidence is stronger for time-series winner legs than short loser legs, and (c) a dynamic, survivorship-free altcoin history was not available here. This is a research decision—not a claim that BTC is guaranteed to remain best.

A demanded 90–100% CAGR is not a valid optimisation objective. At Sharpe 1, roughly 90% expected arithmetic return requires approximately 90% annual risk before volatility drag; with fat tails and estimation error this implies unacceptable ruin risk. Historical early-Bitcoin CAGR cannot be treated as a stationary expectation.

## Evidence reviewed

1. Moskowitz, Ooi & Pedersen document own-return persistence over 1–12 months across 58 liquid equity-index, currency, commodity and bond futures, with diversified crisis performance ([SSRN 2089463](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463)).
2. Hurst, Ooi & Pedersen extend trend evidence to 1880–2016, reporting positive net performance in each decade, but much lower returns in 2010–2016 ([paper](https://static.twentyoverten.com/593e8a9e7299b471eaecf644/SkLoGL67M/A-Century-of-Evidence-on-Trend-Following-Investing.pdf)). This supports persistence, not 100% CAGR.
3. Hamill, Rattray & van Hemert find trend returns positively skewed and strongest in severe equity/bond environments ([SSRN 2831926](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2831926)).
4. Baltas & Kosowski show volatility estimation, trend detection and signed-correlation sizing can cut turnover and improve post-2008 implementation after futures costs ([SSRN 2140091](https://www.ssrn.com/abstract=2140091)).
5. Moreira & Muir report that reducing exposure when volatility is high improves factor Sharpe ratios ([SSRN 2659431](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2659431)); later work and critiques mean this must be treated as a hypothesis, not free alpha.
6. Kim, Tse & Wald show much of famous TSMOM alpha is attributable to volatility scaling and that post-crisis persistence weakens ([SSRN 2786955](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2786955)). This is why the engine reports sizing and raw signal separately.
7. Drogen, Hoffstein & Otte find 30-day winner continuation over the next seven days in digital assets, with long-only momentum exceeding BTC in their sample ([SSRN 4322637](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4322637)).
8. Han, Kang & Ryu find stronger crypto time-series than cross-sectional momentum under realistic costs; short losers often rebound and many levered portfolios liquidate ([SSRN 4675565](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4675565)). This motivates long/flat, leverage caps and jump limits.
9. Crypto survivorship research estimates an enormous 62.19% annualised equal-weight bias versus 0.93% value-weight bias in a survival-conditioned sample ([SSRN 4287573](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4287573)). A present-day altcoin grid is therefore rejected.
10. Bailey et al. introduce CSCV/PBO because ordinary holdouts are unreliable after repeated search ([SSRN 2326253](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)); Bailey & López de Prado's DSR corrects selection and non-normality ([SSRN 2460551](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)).

No independently audited evidence was found that a Jeff Sekinger product sustainably compounds at 90–100% CAGR. Search results were marketing and anecdotal user discussions, not broker statements, a prospectus, independently administered NAV, or peer-reviewed replication. That claim is not used as a research prior.

## Exact model

At daily close, each EMA pair casts one long vote if fast EMA is above slow EMA. The mean of four votes is forecast `f∈[0,1]`. Ex-ante annual volatility is the conservative maximum of a 20-day-span EWMA and a causal rolling GARCH(1,1) forecast. The GARCH model is estimated only on a trailing two-year window after one year of warmup and refitted every 90 days. Before overlays:

`weight = f × target_vol / max(EWMA_vol, GARCH_vol)`.

Using the maximum means GARCH can reduce exposure but cannot increase leverage relative to the fast EWMA model.

It is then limited by:

- 1.25× maximum exposure;
- weight such that a six-sigma daily move loses no more than 20%;
- 50% risk reduction when fast volatility exceeds 1.75× 120-day volatility;
- linear equity drawdown governor from full risk at 15% DD to 25% risk at 30% DD;
- no-trade buffer equal to 10% of current exposure.

The position decided after close `t` earns return `t+1`. Trading cost defaults to 10 bp fee + 5 bp slippage per unit of one-way turnover. Above 1× exposure incurs 6% annual funding. No stop is presumed to fill inside a daily bar.

## Institutional acceptance gates

A strategy is a candidate—not “verified alpha”—only if all are satisfied:

- stitched walk-forward OOS Sharpe > 0.7 and Calmar > 0.7;
- PSR(SR>0) > 95%, with DSR/PBO added when comparing the full strategy library;
- bootstrap 95% CAGR lower bound > 0 after dependent block resampling;
- positive results in at least three materially different market regimes;
- parameter-neighbourhood stability, not one winning cell;
- doubled-cost and one-day additional-delay stress remain profitable;
- no fold dominates lifetime P&L and no single venue is the sole source;
- paper trading plus shadow reconciliation before capital;
- leverage set from loss tolerance, never reverse-engineered to a CAGR target.

## What still needs external infrastructure

For capital deployment replace Yahoo/public CSV with immutable exchange candles and trades from at least two vendors; include timestamp/venue/time-zone policy, missing-bar and outlier quarantine, corporate/contract metadata, fee tiers, spread/impact by order size, funding history, exchange outage models, custody and counterparty limits. Add broker-specific execution with idempotent client IDs, reconciliation, stale-data kill switch, max order/notional limits, circuit breakers, alerts and independent risk sign-off.
