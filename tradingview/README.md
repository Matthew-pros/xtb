# TQQQ TEMACD — ATR Risk Overlay v6

Paste `TQQQ_TEMACD_ATR_v6.pine` into TradingView's Pine Editor and add it to a **daily TQQQ chart**. The supplied TEMACD entry events remain the same; the ATR module is an independent risk overlay.

## EMA baseline versus repository TEMA

The linked Python repository computes three **TEMA** lines with TA-Lib, while the Pine strategy supplied in the conversation computes three ordinary **EMA** lines despite its TEMACD name. The Pine implementation therefore defaults to `EMA (original Pine)` to preserve the exact baseline and offers `TEMA (repository challenger)` as an explicit opt-in. Never splice TEMA results into the EMA baseline: treat them as separate strategy trials when correcting Sharpe/selection statistics.

The repository also combines component entry and exit events with OR, uses long-only `vectorbt.from_signals`, shifts signals by one row, and assumes 10 bp fee plus 10 bp slippage. The Pine version preserves OR events, defaults long-only and uses confirmed bars. Configure commission and slippage explicitly in TradingView Strategy Properties before comparing outputs.

## Default ATR behaviour

1. **Entry regime filter:** blocks new entries if ATR(14) as a percentage of price exceeds 1.5× its 100-bar EMA baseline or 10% absolutely.
2. **Shock exit:** closes an existing position when the same normalized ATR ratio reaches 2.0×.
3. **Ratcheting stop:** initial stop is three ATR from entry and then trails the highest high (long) or lowest low (short); it never loosens.
4. **Risk sizing:** available but off by default. If enabled, quantity is capped to approximately 1% equity risk at the initial stop and can operate only while the ATR stop is enabled.

Orders are generated only on confirmed bars. With TradingView's default next-bar execution, a close signal is filled at the following bar's open in the broker emulator. Stops can gap and fill worse than their stop price. `calc_on_every_tick=false` intentionally prevents a historical/live mismatch from intrabar crossover recalculation.

## Required validation

Do not optimise one best ATR triple and report it. Run a coarse plateau test such as:

- ATR length: 10, 14, 20;
- baseline: 60, 100, 150;
- entry ratio: 1.25, 1.50, 1.75;
- stop: 2.0, 2.5, 3.0, 3.5 ATR;
- shock exit: 1.75, 2.0, 2.5.

Freeze parameters on training data, retain an untouched chronological holdout, and include TradingView commission/slippage in Strategy Properties. Prefer a broad stable neighbourhood over the highest historical CAGR. Compare TEMACD with all ATR switches disabled to isolate the overlay's incremental effect.

TQQQ already targets approximately three times the daily Nasdaq-100 move. Keep the script's additional `Leverage` at 1 unless account-level liquidation and gap risk have been independently modelled. No ATR filter can guarantee lower drawdown or correct fills on live data.
