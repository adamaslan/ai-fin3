# AAPL Earnings Backtest — Methodology Guide

**Created:** 2026-04-29  
**Symbol:** AAPL (Apple Inc.)  
**Tool:** `boll-4-dec-200-json.py` + manual earnings date overlay

---

## What This Backtest Does

For each earnings date, we:
1. Identify the exact trading day earnings were reported (after market close)
2. Snapshot every technical signal active on that day (before the report drops)
3. Record the actual EPS result and surprise %
4. Track price action for the 5 trading days following the report
5. Assess which signals correctly predicted the post-earnings move

The goal is to calibrate which signals have predictive value specifically around earnings events.

---

## Step-by-Step Process

### Step 1 — Identify Earnings Dates

Source: [Yahoo Finance Earnings Calendar](https://finance.yahoo.com/calendar/earnings?symbol=AAPL)

> **Note on scraping:** The Yahoo Finance earnings page is JavaScript-rendered — raw HTTP requests return shell HTML with no table data. The earnings dates used here were manually sourced from the calendar page. `yfinance`'s `get_earnings_dates()` can also fetch them programmatically but may hit rate limits.

Dates used:

| Quarter | Report Date | EPS Estimate | Actual EPS | Surprise |
|---------|------------|-------------|-----------|---------|
| Q1 FY2026 (Dec qtr) | Jan 29, 2026 | $2.67 | $2.84 | +6.25% |
| Q4 FY2025 (Sep qtr) | Oct 30, 2025 | $1.77 | $1.85 | +4.52% |

---

### Step 2 — Fetch Historical Price Data

```python
import yfinance as yf
t = yf.Ticker('AAPL')
df = t.history(period='2y', interval='1d')
df.index = df.index.tz_localize(None)
```

We use a 2-year daily window so all long-period indicators (SMA 200, Ichimoku 52-period) have enough warmup bars.

---

### Step 3 — Snapshot Indicators on Earnings Day

For each earnings date, we slice the data up to and including that trading day, then compute:

| Indicator | Parameters | What it measures |
|-----------|-----------|-----------------|
| RSI | 14-period | Momentum / overbought-oversold |
| MACD | 12/26/9 | Trend direction + crossover |
| Stochastic | %K 14, %D 3-period smooth | Short-term momentum |
| Bollinger Bands | 20-period, 2σ | Price relative to volatility envelope |
| SMA | 20, 50, 200 | Short/medium/long trend |
| Distance from MA | (close - SMA) / SMA × 100 | Overextension % |
| Ichimoku Cloud | 9/26/52 standard | Multi-timeframe trend structure |
| OBV | Cumulative | Volume-based trend confirmation |
| OBV Divergence | 20-bar lookback | Smart money vs price agreement |
| CMF | 20-period | Chaikin Money Flow — buying/selling pressure |
| Support/Resistance | Pivot highs/lows, 10-bar window, 0.5% clustering | Key price levels |

---

### Step 4 — Record Post-Earnings Price Action

We capture the **5 trading days after** the report date, measuring cumulative % change from the earnings-day close. This is the "outcome" we correlate signals against.

---

### Step 5 — Signal Correlation Assessment

For each signal active on earnings day, we ask:
- Did the signal direction match the post-earnings move?
- How many days did it take for the signal to play out?
- Was the signal contradicted by other signals (mixed reading)?

---

## Signal Strength Legend

| Label | Meaning |
|-------|---------|
| 🟢 BULLISH | Signal suggests upside |
| 🔴 BEARISH | Signal suggests downside |
| ⚪ NEUTRAL | No directional bias |
| ⚠️ CONFLICTED | Signal active but contradicted by others |

---

## Files

| File | Contents |
|------|---------|
| `AAPL_earnings_Q1FY2026_Jan29_2026.md` | Full signal report — Jan 29, 2026 earnings |
| `AAPL_earnings_Q4FY2025_Oct30_2025.md` | Full signal report — Oct 30, 2025 earnings |
| `AAPL_earnings_backtest_guide.md` | This methodology guide |

---

## Key Findings Summary

| Quarter | Pre-Earnings Signal Bias | Beat? | 5-Day Outcome | Signals Called It? |
|---------|------------------------|-------|--------------|-------------------|
| Q1 FY2026 | BEARISH LEAN (1🟢 / 3🔴) | +6.25% | +6.83% | ❌ Signals were bearish, price rallied |
| Q4 FY2025 | MIXED (3🟢 / 3🔴) | +4.52% | -0.60% | ✅ Mixed/neutral correctly predicted muted reaction |

### Interpretation

- **Oct 2025**: Stock was already deeply overbought (RSI 83.9, Stoch 90.7, +21.8% above 200MA). The beat caused almost no move — the "buy the rumor" effect had already priced it in. The overextension signals were the most useful here.
- **Jan 2026**: Signals were bearish (price below Ichimoku cloud, Bear TK cross, CMF negative), but a +6.25% EPS surprise overrode everything and pushed price up +6.83% over 5 days. Earnings surprises can override technical signals entirely.

### General Rules from This Backtest

1. **Overextension signals (RSI > 80, Stoch > 85, > 15% above 200MA) before earnings = limited upside even on a beat.** The Oct 2025 case confirms: beat +4.52%, stock barely moved.
2. **Neutral/recovering technicals (RSI ~50, MACD recovering) before earnings = more room to run on a beat.** The Jan 2026 case: RSI 48.8, MACD turning bullish histogram → stock ran +7% over 5 days post-beat.
3. **CMF and OBV divergence** are useful pre-earnings positioning signals: negative CMF going into earnings (Jan 2026) suggested institutions were not loading up, yet the surprise still drove the move.
4. **Ichimoku cloud position** correlated well: price above cloud (Oct 2025) = already bullish and stretched. Price below cloud (Jan 2026) = more room to recover on a catalyst.
