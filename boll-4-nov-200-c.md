# 📊 Enhanced Technical Analysis Scanner
## Complete Documentation & Signal Reference

---

## 🚀 Quick Start

### Installation
```bash
pip install yfinance pandas numpy google-generativeai
```

### Basic Usage
```python
import os
from technical_analyzer import run_analysis

# Simple analysis
analyzer = run_analysis('AAPL', period='1y')

# With AI ranking (recommended)
analyzer = run_analysis(
    symbol='TSLA',
    period='6mo', 
    gemini_api_key=os.getenv('GEMINI_API_KEY')
)
```

---

## 🔄 Main Workflow

### Step 1: Initialize Analyzer
```python
analyzer = TechnicalAnalyzer(
    symbol='AAPL',
    period='1y',
    gemini_api_key='your-api-key'
)
```

### Step 2: Fetch Market Data
- Downloads historical price data from Yahoo Finance
- Includes Open, High, Low, Close, Volume
- Configurable time periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max

### Step 3: Calculate Technical Indicators
Computes 80+ technical indicators including:
- Moving Averages (SMA, EMA)
- Momentum indicators (RSI, MACD, Stochastic)
- Volatility measures (ATR, Bollinger Bands)
- Volume indicators (OBV, MFI, VWAP)
- Trend indicators (ADX, Ichimoku)

### Step 4: Detect Signals
Scans for 150+ technical signals across 25+ categories

### Step 5: AI Ranking
Uses Gemini AI to score each signal from 1-100 based on:
- Actionability
- Reliability
- Timing
- Risk/Reward potential
- Market context

### Step 6: Save & Display Results
- Saves to local timestamped folders
- Exports CSV, JSON, and TXT reports
- Displays formatted results

---

## 📊 Complete Signal Reference

### 1️⃣ Moving Average Signals (MA_CROSS, MA_TREND, MA_COMPRESSION, MA_SLOPE, MA_DISTANCE)

#### **GOLDEN CROSS** 🟢
- **Category:** MA_CROSS
- **Strength:** STRONG BULLISH
- **Description:** 50-day MA crosses above 200-day MA
- **Significance:** Major long-term bullish reversal signal
- **Trading Implication:** Strong buy signal for long-term positions

#### **DEATH CROSS** 🔴
- **Category:** MA_CROSS
- **Strength:** STRONG BEARISH
- **Description:** 50-day MA crosses below 200-day MA
- **Significance:** Major long-term bearish reversal signal
- **Trading Implication:** Strong sell signal or exit positions

#### **PRICE ABOVE 10 MA** 🟢
- **Category:** MA_CROSS
- **Strength:** BULLISH
- **Description:** Price crosses above 10-day moving average
- **Significance:** Short-term bullish momentum
- **Trading Implication:** Potential entry for swing trades

#### **PRICE BELOW 10 MA** 🔴
- **Category:** MA_CROSS
- **Strength:** BEARISH
- **Description:** Price crosses below 10-day moving average
- **Significance:** Short-term bearish momentum
- **Trading Implication:** Consider profit-taking or stop-loss

#### **PRICE ABOVE 20 MA** 🟢
- **Category:** MA_CROSS
- **Strength:** BULLISH
- **Description:** Price crosses above 20-day moving average
- **Significance:** Medium-term bullish trend confirmation
- **Trading Implication:** Good entry point for trend following

#### **PRICE BELOW 20 MA** 🔴
- **Category:** MA_CROSS
- **Strength:** BEARISH
- **Description:** Price crosses below 20-day moving average
- **Significance:** Medium-term bearish trend confirmation
- **Trading Implication:** Exit long positions or consider shorts

#### **10/20 EMA BULL CROSS** 🟢
- **Category:** MA_CROSS
- **Strength:** BULLISH
- **Description:** 10-day EMA crosses above 20-day EMA
- **Significance:** Short-term momentum shift upward
- **Trading Implication:** Early entry signal for aggressive traders

#### **10/20 EMA BEAR CROSS** 🔴
- **Category:** MA_CROSS
- **Strength:** BEARISH
- **Description:** 10-day EMA crosses below 20-day EMA
- **Significance:** Short-term momentum shift downward
- **Trading Implication:** Early exit signal

#### **MA ALIGNMENT BULLISH** 🟢
- **Category:** MA_TREND
- **Strength:** STRONG BULLISH
- **Description:** 10 SMA > 20 SMA > 50 SMA (perfect alignment)
- **Significance:** Strong uptrend with all timeframes aligned
- **Trading Implication:** High probability continuation trade

#### **MA ALIGNMENT BEARISH** 🔴
- **Category:** MA_TREND
- **Strength:** STRONG BEARISH
- **Description:** 10 SMA < 20 SMA < 50 SMA (perfect alignment)
- **Significance:** Strong downtrend with all timeframes aligned
- **Trading Implication:** Avoid longs, consider shorts

#### **ABOVE 200 SMA** 🟢
- **Category:** MA_TREND
- **Strength:** BULLISH
- **Description:** Price above 200-day moving average
- **Significance:** Long-term uptrend intact
- **Trading Implication:** Bias toward long positions

#### **BELOW 200 SMA** 🔴
- **Category:** MA_TREND
- **Strength:** BEARISH
- **Description:** Price below 200-day moving average
- **Significance:** Long-term downtrend intact
- **Trading Implication:** Bias toward short positions

#### **MA COMPRESSION** 🟡
- **Category:** MA_COMPRESSION
- **Strength:** NEUTRAL
- **Description:** Moving averages converging within 2%
- **Significance:** Breakout imminent
- **Trading Implication:** Prepare for directional move

#### **MA SLOPE REVERSAL UP** 🟢
- **Category:** MA_SLOPE
- **Strength:** BULLISH
- **Description:** 20-day MA changing from downward to upward slope
- **Significance:** Trend reversal in progress
- **Trading Implication:** Early trend change signal

#### **MA SLOPE REVERSAL DOWN** 🔴
- **Category:** MA_SLOPE
- **Strength:** BEARISH
- **Description:** 20-day MA changing from upward to downward slope
- **Significance:** Trend reversal in progress
- **Trading Implication:** Warning of trend deterioration

#### **OVEREXTENDED ABOVE 20MA** 🔴
- **Category:** MA_DISTANCE
- **Strength:** BEARISH
- **Description:** Price >10% above 20-day MA
- **Significance:** Potential pullback zone
- **Trading Implication:** Take profits, wait for retest

#### **OVEREXTENDED BELOW 20MA** 🟢
- **Category:** MA_DISTANCE
- **Strength:** BULLISH
- **Description:** Price >10% below 20-day MA
- **Significance:** Potential bounce zone
- **Trading Implication:** Look for reversal entry

---

### 2️⃣ RSI Signals (RSI, RSI_MOMENTUM, RSI_DIVERGENCE, RSI_CROSS)

#### **RSI OVERSOLD** 🟢
- **Category:** RSI
- **Strength:** BULLISH
- **Description:** RSI below 30
- **Significance:** Potential reversal zone
- **Trading Implication:** Watch for bullish confirmation

#### **RSI OVERBOUGHT** 🔴
- **Category:** RSI
- **Strength:** BEARISH
- **Description:** RSI above 70
- **Significance:** Potential reversal zone
- **Trading Implication:** Consider profit-taking

#### **RSI EXTREME OVERSOLD** 🟢🟢
- **Category:** RSI
- **Strength:** STRONG BULLISH
- **Description:** RSI below 20
- **Significance:** Extremely oversold, high reversal probability
- **Trading Implication:** Strong buy opportunity

#### **RSI EXTREME OVERBOUGHT** 🔴🔴
- **Category:** RSI
- **Strength:** STRONG BEARISH
- **Description:** RSI above 80
- **Significance:** Extremely overbought, high reversal probability
- **Trading Implication:** Strong sell signal

#### **RSI BULLISH DIVERGENCE** 🟢
- **Category:** RSI_DIVERGENCE
- **Strength:** BULLISH
- **Description:** Price making lower lows, RSI making higher lows
- **Significance:** Momentum shifting despite price weakness
- **Trading Implication:** Potential trend reversal to upside

#### **RSI BEARISH DIVERGENCE** 🔴
- **Category:** RSI_DIVERGENCE
- **Strength:** BEARISH
- **Description:** Price making higher highs, RSI making lower highs
- **Significance:** Momentum waning despite price strength
- **Trading Implication:** Potential trend reversal to downside

#### **RSI MOMENTUM SURGE** 🟢
- **Category:** RSI_MOMENTUM
- **Strength:** BULLISH
- **Description:** RSI increased >10 points in 5 days while below 50
- **Significance:** Accelerating bullish momentum
- **Trading Implication:** Ride the momentum

#### **RSI MOMENTUM COLLAPSE** 🔴
- **Category:** RSI_MOMENTUM
- **Strength:** BEARISH
- **Description:** RSI decreased >10 points in 5 days while above 50
- **Significance:** Accelerating bearish momentum
- **Trading Implication:** Exit positions quickly

#### **RSI ABOVE 50** 🟢
- **Category:** RSI_CROSS
- **Strength:** BULLISH
- **Description:** RSI crossed above 50 centerline
- **Significance:** Entering bullish zone
- **Trading Implication:** Confirmation of uptrend

#### **RSI BELOW 50** 🔴
- **Category:** RSI_CROSS
- **Strength:** BEARISH
- **Description:** RSI crossed below 50 centerline
- **Significance:** Entering bearish zone
- **Trading Implication:** Confirmation of downtrend

---

### 3️⃣ MACD Signals (MACD, MACD_MOMENTUM)

#### **MACD BULL CROSS** 🟢
- **Category:** MACD
- **Strength:** BULLISH
- **Description:** MACD line crosses above signal line
- **Significance:** Momentum shifting bullish
- **Trading Implication:** Buy signal

#### **MACD BEAR CROSS** 🔴
- **Category:** MACD
- **Strength:** BEARISH
- **Description:** MACD line crosses below signal line
- **Significance:** Momentum shifting bearish
- **Trading Implication:** Sell signal

#### **MACD ABOVE ZERO** 🟢
- **Category:** MACD
- **Strength:** BULLISH
- **Description:** MACD crossed into positive territory
- **Significance:** Bullish momentum confirmed
- **Trading Implication:** Trend strength confirmation

#### **MACD BELOW ZERO** 🔴
- **Category:** MACD
- **Strength:** BEARISH
- **Description:** MACD crossed into negative territory
- **Significance:** Bearish momentum confirmed
- **Trading Implication:** Downtrend confirmation

#### **MACD HISTOGRAM EXPANSION** (Bullish) 🟢
- **Category:** MACD_MOMENTUM
- **Strength:** STRONG BULLISH
- **Description:** Histogram bars expanding while positive
- **Significance:** Accelerating bullish momentum
- **Trading Implication:** Strong continuation signal

#### **MACD HISTOGRAM EXPANSION** (Bearish) 🔴
- **Category:** MACD_MOMENTUM
- **Strength:** STRONG BEARISH
- **Description:** Histogram bars expanding while negative
- **Significance:** Accelerating bearish momentum
- **Trading Implication:** Strong continuation signal

---

### 4️⃣ Bollinger Bands (BOLLINGER, BB_SQUEEZE, BB_WALK, BB_EXTREME)

#### **BB SQUEEZE** 🟡
- **Category:** BB_SQUEEZE
- **Strength:** NEUTRAL
- **Description:** Bands narrowing <70% of average width
- **Significance:** Volatility contraction, breakout pending
- **Trading Implication:** Prepare for large move in either direction

#### **AT LOWER BB** 🟢
- **Category:** BOLLINGER
- **Strength:** BULLISH
- **Description:** Price touching lower Bollinger Band
- **Significance:** Potential oversold bounce
- **Trading Implication:** Look for reversal signals

#### **AT UPPER BB** 🔴
- **Category:** BOLLINGER
- **Strength:** BEARISH
- **Description:** Price touching upper Bollinger Band
- **Significance:** Potential overbought reversal
- **Trading Implication:** Consider profit-taking

#### **BB WALK UPPER** 🟢🟢
- **Category:** BB_WALK
- **Strength:** STRONG BULLISH
- **Description:** Price riding upper band for 3+ days
- **Significance:** Very strong uptrend
- **Trading Implication:** Let winners run, trail stops

#### **BB WALK LOWER** 🔴🔴
- **Category:** BB_WALK
- **Strength:** STRONG BEARISH
- **Description:** Price riding lower band for 3+ days
- **Significance:** Very strong downtrend
- **Trading Implication:** Avoid catching falling knife

#### **ABOVE UPPER BB** 🟢🟢🟢
- **Category:** BB_EXTREME
- **Strength:** EXTREME BULLISH
- **Description:** Price extended beyond upper band
- **Significance:** Extreme strength or potential exhaustion
- **Trading Implication:** Trail stops, prepare for pullback

#### **BELOW LOWER BB** 🔴🔴🔴
- **Category:** BB_EXTREME
- **Strength:** EXTREME BEARISH
- **Description:** Price extended beyond lower band
- **Significance:** Extreme weakness or potential reversal
- **Trading Implication:** Wait for stabilization

---

### 5️⃣ Volume Signals (VOLUME, VOLUME_DIVERGENCE, VOLUME_ACCUMULATION, VOLUME_CLIMAX)

#### **VOLUME SPIKE 2X** ⚡
- **Category:** VOLUME
- **Strength:** SIGNIFICANT
- **Description:** Volume >200% of 20-day average
- **Significance:** Unusual interest/activity
- **Trading Implication:** Pay attention to price direction

#### **EXTREME VOLUME 3X** ⚡⚡
- **Category:** VOLUME
- **Strength:** VERY SIGNIFICANT
- **Description:** Volume >300% of 20-day average
- **Significance:** Major event or institutional activity
- **Trading Implication:** Expect large moves

#### **VOLUME BREAKOUT** 🟢
- **Category:** VOLUME
- **Strength:** STRONG BULLISH
- **Description:** High volume + price up >2%
- **Significance:** Strong buying pressure
- **Trading Implication:** Breakout confirmation

#### **VOLUME SELLOFF** 🔴
- **Category:** VOLUME
- **Strength:** STRONG BEARISH
- **Description:** High volume + price down >2%
- **Significance:** Strong selling pressure
- **Trading Implication:** Breakdown confirmation

#### **VOLUME DIVERGENCE BEARISH** 🔴
- **Category:** VOLUME_DIVERGENCE
- **Strength:** BEARISH
- **Description:** Price rising but volume declining
- **Significance:** Weakening uptrend, lack of conviction
- **Trading Implication:** Warning of potential reversal

#### **VOLUME DIVERGENCE BULLISH** 🟢
- **Category:** VOLUME_DIVERGENCE
- **Strength:** BULLISH
- **Description:** Price falling but volume declining
- **Significance:** Selling pressure exhausting
- **Trading Implication:** Potential bottom forming

#### **VOLUME ACCUMULATION** 🟢
- **Category:** VOLUME_ACCUMULATION
- **Strength:** STRONG BULLISH
- **Description:** Consistent volume increase + price increase
- **Significance:** Strong accumulation phase
- **Trading Implication:** Join the trend

#### **SELLING CLIMAX** 🔴🔴🔴
- **Category:** VOLUME_CLIMAX
- **Strength:** EXTREME
- **Description:** Volume >400% average with price decline
- **Significance:** Panic selling, potential capitulation
- **Trading Implication:** Often marks short-term bottom

#### **BUYING CLIMAX** 🟢🟢🟢
- **Category:** VOLUME_CLIMAX
- **Strength:** EXTREME
- **Description:** Volume >400% average with price increase
- **Significance:** Euphoric buying, potential exhaustion
- **Trading Implication:** Often marks short-term top

---

### 6️⃣ Price Action (PRICE_ACTION, PRICE_PATTERN, GAP, CANDLESTICK)

#### **LARGE GAIN** 🟢
- **Category:** PRICE_ACTION
- **Strength:** STRONG BULLISH
- **Description:** Single-day gain >5%
- **Significance:** Strong momentum day
- **Trading Implication:** Momentum likely continues

#### **LARGE LOSS** 🔴
- **Category:** PRICE_ACTION
- **Strength:** STRONG BEARISH
- **Description:** Single-day loss >5%
- **Significance:** Strong selling pressure
- **Trading Implication:** Momentum likely continues

#### **52-WEEK HIGH** 🟢🟢
- **Category:** RANGE
- **Strength:** STRONG BULLISH
- **Description:** Price at or near 52-week high
- **Significance:** Breakout to new highs
- **Trading Implication:** Uptrend strength, new territory

#### **52-WEEK LOW** 🔴🔴
- **Category:** RANGE
- **Strength:** STRONG BEARISH
- **Description:** Price at or near 52-week low
- **Significance:** Breakdown to new lows
- **Trading Implication:** Downtrend strength, capitulation zone

#### **HIGHER HIGHS & LOWS** 🟢
- **Category:** PRICE_PATTERN
- **Strength:** STRONG BULLISH
- **Description:** Clear pattern of ascending highs and lows
- **Significance:** Textbook uptrend
- **Trading Implication:** Buy dips, ride the trend

#### **LOWER HIGHS & LOWS** 🔴
- **Category:** PRICE_PATTERN
- **Strength:** STRONG BEARISH
- **Description:** Clear pattern of descending highs and lows
- **Significance:** Textbook downtrend
- **Trading Implication:** Sell rallies, stay short

#### **GAP UP** 🟢
- **Category:** GAP
- **Strength:** BULLISH
- **Description:** Opening >2% above previous close
- **Significance:** Overnight bullish news/sentiment
- **Trading Implication:** Strong opening often continues

#### **GAP DOWN** 🔴
- **Category:** GAP
- **Strength:** BEARISH
- **Description:** Opening >2% below previous close
- **Significance:** Overnight bearish news/sentiment
- **Trading Implication:** Weak opening often continues

#### **INSIDE BAR** 🟡
- **Category:** PRICE_PATTERN
- **Strength:** NEUTRAL
- **Description:** Current bar contained within previous bar's range
- **Significance:** Consolidation, compression
- **Trading Implication:** Breakout setup, wait for direction

#### **OUTSIDE BAR** ⚡
- **Category:** PRICE_PATTERN
- **Strength:** SIGNIFICANT
- **Description:** Current bar engulfs previous bar's range
- **Significance:** Volatility expansion
- **Trading Implication:** Directional move in progress

#### **BULLISH ENGULFING** 🟢
- **Category:** CANDLESTICK
- **Strength:** STRONG BULLISH
- **Description:** Bullish candle completely engulfs previous bearish candle
- **Significance:** Strong reversal pattern
- **Trading Implication:** Buy signal at support

#### **BEARISH ENGULFING** 🔴
- **Category:** CANDLESTICK
- **Strength:** STRONG BEARISH
- **Description:** Bearish candle completely engulfs previous bullish candle
- **Significance:** Strong reversal pattern
- **Trading Implication:** Sell signal at resistance

#### **DOJI CANDLE** 🟡
- **Category:** CANDLESTICK
- **Strength:** NEUTRAL
- **Description:** Open and close nearly equal (small body)
- **Significance:** Indecision in market
- **Trading Implication:** Potential reversal, wait for confirmation

#### **HAMMER PATTERN** 🟢
- **Category:** CANDLESTICK
- **Strength:** BULLISH
- **Description:** Small body with long lower wick (2:1 ratio)
- **Significance:** Buyers rejected lower prices
- **Trading Implication:** Reversal signal at bottom

#### **SHOOTING STAR** 🔴
- **Category:** CANDLESTICK
- **Strength:** BEARISH
- **Description:** Small body with long upper wick (2:1 ratio)
- **Significance:** Sellers rejected higher prices
- **Trading Implication:** Reversal signal at top

---

### 7️⃣ Support & Resistance (PIVOT, FIBONACCI, STRUCTURE_BREAK)

#### **ABOVE R1 PIVOT** 🟢
- **Category:** PIVOT
- **Strength:** BULLISH
- **Description:** Price broke above first resistance pivot
- **Significance:** Resistance becomes support
- **Trading Implication:** Continuation likely to R2

#### **BELOW S1 PIVOT** 🔴
- **Category:** PIVOT
- **Strength:** BEARISH
- **Description:** Price broke below first support pivot
- **Significance:** Support becomes resistance
- **Trading Implication:** Continuation likely to S2

#### **AT FIB 61.8%** ⚡
- **Category:** FIBONACCI
- **Strength:** SIGNIFICANT
- **Description:** Price at golden ratio retracement level
- **Significance:** Key decision point
- **Trading Implication:** Major support/resistance zone

#### **AT FIB 50%** ⚡
- **Category:** FIBONACCI
- **Strength:** SIGNIFICANT
- **Description:** Price at 50% retracement level
- **Significance:** Psychological midpoint
- **Trading Implication:** Important support/resistance

#### **RESISTANCE BREAKOUT** 🟢
- **Category:** STRUCTURE_BREAK
- **Strength:** STRONG BULLISH
- **Description:** Price broke above 20-day resistance level
- **Significance:** New uptrend phase beginning
- **Trading Implication:** Strong buy signal

#### **SUPPORT BREAKDOWN** 🔴
- **Category:** STRUCTURE_BREAK
- **Strength:** STRONG BEARISH
- **Description:** Price broke below 20-day support level
- **Significance:** New downtrend phase beginning
- **Trading Implication:** Strong sell signal

---

### 8️⃣ Oscillators (STOCHASTIC, WILLIAMS_R, CCI, MFI, ADX, OBV)

#### **STOCHASTIC OVERSOLD** 🟢
- **Category:** STOCHASTIC
- **Strength:** BULLISH
- **Description:** Stochastic K below 20
- **Significance:** Short-term oversold
- **Trading Implication:** Reversal potential

#### **STOCHASTIC OVERBOUGHT** 🔴
- **Category:** STOCHASTIC
- **Strength:** BEARISH
- **Description:** Stochastic K above 80
- **Significance:** Short-term overbought
- **Trading Implication:** Reversal potential

#### **STOCHASTIC BULL CROSS** 🟢
- **Category:** STOCHASTIC
- **Strength:** BULLISH
- **Description:** %K crossed above %D
- **Significance:** Momentum turning up
- **Trading Implication:** Buy signal

#### **STOCHASTIC BEAR CROSS** 🔴
- **Category:** STOCHASTIC
- **Strength:** BEARISH
- **Description:** %K crossed below %D
- **Significance:** Momentum turning down
- **Trading Implication:** Sell signal

#### **WILLIAMS R OVERSOLD** 🟢
- **Category:** WILLIAMS_R
- **Strength:** BULLISH
- **Description:** Williams %R below -80
- **Significance:** Deeply oversold
- **Trading Implication:** Bounce likely

#### **WILLIAMS R OVERBOUGHT** 🔴
- **Category:** WILLIAMS_R
- **Strength:** BEARISH
- **Description:** Williams %R above -20
- **Significance:** Deeply overbought
- **Trading Implication:** Pullback likely

#### **CCI OVERBOUGHT** 🔴
- **Category:** CCI
- **Strength:** BEARISH
- **Description:** Commodity Channel Index >100
- **Significance:** Extended beyond typical range
- **Trading Implication:** Mean reversion likely

#### **CCI OVERSOLD** 🟢
- **Category:** CCI
- **Strength:** BULLISH
- **Description:** Commodity Channel Index <-100
- **Significance:** Extended beyond typical range
- **Trading Implication:** Mean reversion likely

#### **MFI OVERSOLD** 🟢
- **Category:** MFI
- **Strength:** BULLISH
- **Description:** Money Flow Index <20
- **Significance:** Selling exhaustion
- **Trading Implication:** Money may start flowing in

#### **MFI OVERBOUGHT** 🔴
- **Category:** MFI
- **Strength:** BEARISH
- **Description:** Money Flow Index >80
- **Significance:** Buying exhaustion
- **Trading Implication:** Money may start flowing out

#### **STRONG UPTREND** / **STRONG DOWNTREND** 🟢/🔴
- **Category:** TREND
- **Strength:** TRENDING
- **Description:** ADX >25
- **Significance:** Strong directional movement
- **Trading Implication:** Trend following strategies work

#### **VERY STRONG TREND** ⚡⚡
- **Category:** TREND
- **Strength:** EXTREME
- **Description:** ADX >40
- **Significance:** Extreme trend strength
- **Trading Implication:** Major trending environment

#### **STRONG BULL TREND ADX** 🟢
- **Category:** ADX
- **Strength:** STRONG BULLISH
- **Description:** ADX >30 with +DI > -DI
- **Significance:** Confirmed strong uptrend
- **Trading Implication:** Buy dips, avoid shorts

#### **STRONG BEAR TREND ADX** 🔴
- **Category:** ADX
- **Strength:** STRONG BEARISH
- **Description:** ADX >30 with -DI > +DI
- **Significance:** Confirmed strong downtrend
- **Trading Implication:** Sell rallies, avoid longs

#### **DI BULL CROSS** 🟢
- **Category:** ADX
- **Strength:** BULLISH
- **Description:** +DI crossed above -DI
- **Significance:** Directional momentum shifting up
- **Trading Implication:** Early trend signal

#### **DI BEAR CROSS** 🔴
- **Category:** ADX
- **Strength:** BEARISH
- **Description:** -DI crossed above +DI
- **Significance:** Directional momentum shifting down
- **Trading Implication:** Early trend deterioration

#### **OBV CONFIRMATION** 🟢/🔴
- **Category:** OBV
- **Strength:** STRONG BULLISH/BEARISH
- **Description:** On-Balance Volume trend matches price trend
- **Significance:** Volume confirming price move
- **Trading Implication:** High confidence trend

---

### 9️⃣ Advanced Indicators (ICHIMOKU, ROC, VWAP, ATR, EXHAUSTION)

#### **ABOVE ICHIMOKU CLOUD** 🟢
- **Category:** ICHIMOKU
- **Strength:** BULLISH
- **Description:** Price above both Senkou A and B
- **Significance:** Bullish cloud position
- **Trading Implication:** Long bias

#### **BELOW ICHIMOKU CLOUD** 🔴
- **Category:** ICHIMOKU
- **Strength:** BEARISH
- **Description:** Price below both Senkou A and B
- **Significance:** Bearish cloud position
- **Trading Implication:** Short bias

#### **TENKAN CROSS UP** 🟢
- **Category:** ICHIMOKU
- **Strength:** BULLISH
- **Description:** Price crossed above Tenkan-sen (conversion line)
- **Significance:** Short-term momentum shift
- **Trading Implication:** Quick trade signal

#### **EXTREME ROC POSITIVE** 🟢
- **Category:** ROC
- **Strength:** STRONG BULLISH
- **Description:** 20-day Rate of Change >15%
- **Significance:** Strong momentum burst
- **Trading Implication:** Ride or take profits

#### **EXTREME ROC NEGATIVE** 🔴
- **Category:** ROC
- **Strength:** STRONG BEARISH
- **Description:** 20-day Rate of Change <-15%
- **Significance:** Sharp decline
- **Trading Implication:** Capitulation or continuation

#### **ABOVE VWAP** 🟢
- **Category:** VWAP
- **Strength:** BULLISH
- **Description:** Price crossed above Volume-Weighted Average Price
- **Significance:** Institutional buying support
- **Trading Implication:** Intraday long bias

#### **BELOW VWAP** 🔴
- **Category:** VWAP
- **Strength:** BEARISH
- **Description:** Price crossed below Volume-Weighted Average Price
- **Significance:** Institutional selling pressure
- **Trading Implication:** Intraday short bias

#### **ATR STOP LEVEL** 🟡
- **Category:** ATR
- **Strength:** NEUTRAL
- **Description:** Calculated stop-loss based on 2x ATR
- **Significance:** Volatility-adjusted risk management
- **Trading Implication:** Recommended stop placement

#### **BULLISH EXHAUSTION** 🔴
- **Category:** EXHAUSTION
- **Strength:** BEARISH
- **Description:** RSI >75 + Volume >2x average
- **Significance:** Buying climax, potential reversal
- **Trading Implication:** Take profits, reversal watch

#### **BEARISH EXHAUSTION** 🟢
- **Category:** EXHAUSTION
- **Strength:** BULLISH
- **Description:** RSI <25 + Volume >2x average
- **Significance:** Selling climax, potential reversal
- **Trading Implication:** Bottom fishing opportunity

---

### 🔟 Multi-Factor Signals (TIMEFRAME_ALIGNMENT, MULTI_INDICATOR, CONSOLIDATION)

#### **FULL TIMEFRAME ALIGNMENT** (Bullish) 🟢🟢🟢
- **Category:** TIMEFRAME_ALIGNMENT
- **Strength:** EXTREME BULLISH
- **Description:** Price above 10, 50, and 200 SMAs
- **Significance:** All timeframes in agreement
- **Trading Implication:** Highest probability long setup

#### **FULL TIMEFRAME ALIGNMENT** (Bearish) 🔴🔴🔴
- **Category:** TIMEFRAME_ALIGNMENT
- **Strength:** EXTREME BEARISH
- **Description:** Price below 10, 50, and 200 SMAs
- **Significance:** All timeframes in agreement
- **Trading Implication:** Highest probability short setup

#### **MULTI-INDICATOR BULLISH** 🟢
- **Category:** MULTI_INDICATOR
- **Strength:** STRONG BULLISH
- **Description:** 4-5 of 5 key indicators bullish (RSI>50, MACD>Signal, Price>SMA20, Stoch K>D, +DI>-DI)
- **Significance:** Broad confirmation of uptrend
- **Trading Implication:** High confidence long entry

#### **MULTI-INDICATOR BEARISH** 🔴
- **Category:** MULTI_INDICATOR
- **Strength:** STRONG BEARISH
- **Description:** 4-5 of 5 key indicators bearish
- **Significance:** Broad confirmation of downtrend
- **Trading Implication:** High confidence short entry

#### **TIGHT CONSOLIDATION** 🟡
- **Category:** CONSOLIDATION
- **Strength:** NEUTRAL
- **Description:** 10-day price range <5% of average price
- **Significance:** Compression phase before expansion
- **Trading Implication:** Breakout setup forming

---

### 1️⃣1️⃣ Momentum Signals (MOMENTUM_DIVERGENCE, MOMENTUM_ACCELERATION)

#### **MOMENTUM DIVERGENCE** 🔴
- **Category:** MOMENTUM_DIVERGENCE
- **Strength:** BEARISH
- **Description:** Price up >5% but volume down >20%
- **Significance:** Move lacks conviction
- **Trading Implication:** Unsustainable rally

#### **MOMENTUM ACCELERATION** 🟢
- **Category:** MOMENTUM_ACCELERATION
- **Strength:** STRONG BULLISH
- **Description:** Momentum increasing for 3+ consecutive periods
- **Significance:** Accelerating uptrend
- **Trading Implication:** Trend strengthening

#### **MOMENTUM DECELERATION** 🔴
- **Category:** MOMENTUM_ACCELERATION
- **Strength:** STRONG BEARISH
- **Description:** Negative momentum increasing for 3+ consecutive periods
- **Significance:** Accelerating downtrend
- **Trading Implication:** Trend strengthening (down)

---

### 1️⃣2️⃣ Volatility Signals (VOLATILITY, VOLATILITY_SQUEEZE)

#### **HIGH VOLATILITY** ⚡
- **Category:** VOLATILITY
- **Strength:** SIGNIFICANT
- **Description:** Annualized volatility >40%
- **Significance:** Elevated risk/opportunity
- **Trading Implication:** Wider stops, smaller positions

#### **LOW VOLATILITY** 🟡
- **Category:** VOLATILITY_SQUEEZE
- **Strength:** NEUTRAL
- **Description:** Annualized volatility <15%
- **Significance:** Compression before expansion
- **Trading Implication:** Prepare for breakout

#### **VOLATILITY EXPANSION** ⚡
- **Category:** VOLATILITY
- **Strength:** SIGNIFICANT
- **Description:** Volatility increased >50% in 10 days
- **Significance:** Market regime change
- **Trading Implication:** Adjust risk management

---

## 📈 Helper Functions

### Access Results
```python
# Get technical data DataFrame
technical_data = analyzer.data
print(technical_data.tail())

# Get signals list
signals = analyzer.signals
print(f"Total signals: {len(signals)}")

# Create DataFrame
signals_df = create_signals_dataframe(analyzer)
print(signals_df.head(10))
```

### Analysis Functions
```python
# Category breakdown
categories = get_category_breakdown(analyzer)

# Strength analysis
strengths = get_strength_analysis(analyzer)

# Export to CSV
export_to_csv(analyzer)
```

### Filter Signals
```python
# High-confidence signals (AI score ≥80)
high_conf = [s for s in analyzer.signals if s.get('ai_score', 0) >= 80]

# Bullish signals only
bullish = [s for s in analyzer.signals if 'BULLISH' in s['strength']]

# Specific category
ma_signals = [s for s in analyzer.signals if 'MA' in s['category']]

# Top 10 ranked
top_10 = analyzer.signals[:10]
```

---

## 🎯 Example Workflows

### Workflow 1: Quick Scan
```python
import os
from technical_analyzer import run_analysis

# Run with AI
analyzer = run_analysis(
    symbol='AAPL',
    period='6mo',
    gemini_api_key=os.getenv('GEMINI_API_KEY')
)

# Review top signals
for sig in analyzer.signals[:5]:
    print(f"{sig['signal']}: {sig['desc']}")
```

### Workflow 2: Detailed Analysis
```python
# Run analysis
analyzer = run_analysis('TSLA', period='1y', gemini_api_key='your-key')

# Create DataFrame
signals_df = create_signals_dataframe(analyzer)

# Filter actionable signals
actionable = signals_df[signals_df['Score'] >= 70]
print(f"\nActionable signals: {len(actionable)}")
print(actionable)

# Category breakdown
categories = get_category_breakdown(analyzer)

# Export
export_to_csv(analyzer)
```

### Workflow 3: Multi-Stock Comparison
```python
symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
results = {}

for symbol in symbols:
    analyzer = run_analysis(symbol, period='3mo', gemini_api_key=api_key)
    
    # Store key metrics
    results[symbol] = {
        'total_signals': len(analyzer.signals),
        'bullish': sum(1 for s in analyzer.signals if 'BULLISH' in s['strength']),
        'bearish': sum(1 for s in analyzer.signals if 'BEARISH' in s['strength']),
        'top_signal': analyzer.signals[0]['signal'] if analyzer.signals else 'None',
        'top_score': analyzer.signals[0].get('ai_score', 0) if analyzer.signals else 0
    }

# Compare results
import pandas as pd
comparison_df = pd.DataFrame(results).T
print(comparison_df.sort_values('top_score', ascending=False))
```

### Workflow 4: Strategy Backtesting Setup
```python
# Get historical data with all indicators
analyzer = run_analysis('SPY', period='2y', gemini_api_key=api_key)
data = analyzer.data

# Example: Test MA crossover strategy
data['Position'] = 0
data.loc[data['SMA_10'] > data['SMA_20'], 'Position'] = 1
data.loc[data['SMA_10'] < data['SMA_20'], 'Position'] = -1

# Calculate returns
data['Returns'] = data['Close'].pct_change()
data['Strategy_Returns'] = data['Position'].shift(1) * data['Returns']

# Performance
cumulative_returns = (1 + data['Strategy_Returns']).cumprod()
print(f"Total Return: {(cumulative_returns.iloc[-1] - 1) * 100:.2f}%")
```

---

## 🔧 Configuration Options

### Time Periods
- `'1d'` - 1 day (intraday)
- `'5d'` - 5 days
- `'1mo'` - 1 month
- `'3mo'` - 3 months (default for swing trading)
- `'6mo'` - 6 months
- `'1y'` - 1 year (comprehensive)
- `'2y'` - 2 years
- `'5y'` - 5 years
- `'max'` - Maximum available

### Local Storage
Files are automatically saved to:
```
technical_analysis_data/
└── YYYY-MM-DD/
    ├── YYYY-MM-DD-SYMBOL-technical_data-HHMMSS.csv
    ├── YYYY-MM-DD-SYMBOL-signals-HHMMSS.json
    └── YYYY-MM-DD-SYMBOL-ranked_signals-HHMMSS.txt
```

---

## 🤖 AI Scoring System

### Score Ranges
- **90-100:** Exceptional signal, high conviction
- **80-89:** Strong signal, good risk/reward
- **70-79:** Solid signal, worth consideration
- **60-69:** Moderate signal, context-dependent
- **50-59:** Weak signal, needs confirmation
- **Below 50:** Low confidence, avoid

### Scoring Factors
1. **Actionability** - How tradeable is this signal?
2. **Reliability** - Historical success rate
3. **Timing** - Is this the right moment?
4. **Risk/Reward** - Potential profit vs. loss
5. **Market Context** - Does it fit current conditions?

### AI Reasoning
Each signal includes AI reasoning explaining:
- Why the score was assigned
- Key factors considered
- Potential risks
- Suggested action

---

## 📊 Technical Indicators Calculated

### Moving Averages
- SMA: 5, 10, 20, 50, 100, 200 period
- EMA: 5, 10, 20, 50, 100, 200 period
- MA slopes and distances

### Momentum Indicators
- RSI (14 period)
- MACD (12, 26, 9)
- Stochastic (14, 3)
- Williams %R (14)
- CCI (20)
- ROC (5, 10, 20)
- Momentum (10)

### Volatility Indicators
- ATR (14)
- Bollinger Bands (20, 2)
- Historical Volatility (20)

### Volume Indicators
- OBV (On-Balance Volume)
- MFI (Money Flow Index)
- VWAP
- Volume Moving Averages (20, 50)

### Trend Indicators
- ADX (14)
- +DI / -DI
- Ichimoku Cloud (9, 26, 52)

### Support/Resistance
- Pivot Points (Traditional)
- Fibonacci Retracement (38.2%, 50%, 61.8%)
- 20-day and 52-week highs/lows

---

## 🎓 Trading Signal Interpretation

### Signal Strength Guide

**EXTREME BULLISH/BEARISH** 🔥🔥🔥
- Rare, high-conviction signals
- Multiple confirming factors
- Clear directional bias
- Actionable for position sizing up

**STRONG BULLISH/BEARISH** 🔥🔥
- High probability setups
- Good risk/reward
- Core trading signals
- Primary entry/exit points

**BULLISH/BEARISH** 🔥
- Standard signals
- Need context confirmation
- Use with other indicators
- Supporting signals

**NEUTRAL/SIGNIFICANT** 🟡
- Informational signals
- Setup indicators
- Prepare for action
- Wait for directional confirmation

### Combining Signals

**Best Practices:**
1. Look for signal clusters (3+ confirming signals)
2. Combine different categories (MA + RSI + Volume)
3. Weight by AI score
4. Consider timeframe alignment
5. Respect overall market trend

**Example Strong Setup:**
- MA Alignment Bullish (STRONG BULLISH)
- RSI Above 50 (BULLISH)
- MACD Bull Cross (BULLISH)
- Volume Breakout (STRONG BULLISH)
- Multi-Indicator Bullish (STRONG BULLISH)

**Result:** High confidence long entry with 5 confirming signals

---

## ⚠️ Risk Management

### Stop Loss Recommendations
```python
# ATR-based stops
current = analyzer.data.iloc[-1]
atr_stop = current['ATR'] * 2

if going_long:
    stop_loss = current['Close'] - atr_stop
else:
    stop_loss = current['Close'] + atr_stop
```

### Position Sizing
```python
# Risk-based sizing
account_size = 100000
risk_per_trade = 0.02  # 2%
risk_amount = account_size * risk_per_trade

entry = current['Close']
stop = entry - (current['ATR'] * 2)
risk_per_share = abs(entry - stop)

position_size = int(risk_amount / risk_per_share)
```

### Signal Filtering by Volatility
```python
# Adjust strategy based on volatility regime
current_vol = analyzer.data.iloc[-1]['Volatility']

if current_vol > 40:
    # High volatility: Reduce size, widen stops
    position_multiplier = 0.5
    atr_multiplier = 3
elif current_vol < 15:
    # Low volatility: Normal size, tighter stops
    position_multiplier = 1.0
    atr_multiplier = 1.5
else:
    # Normal volatility
    position_multiplier = 1.0
    atr_multiplier = 2.0
```

---

## 📚 Additional Resources

### Get Gemini API Key
Visit: [Google AI Studio](https://aistudio.google.com/app/apikey)

Set environment variable:
```bash
export GEMINI_API_KEY='your-api-key-here'
```

### Data Source
Market data powered by [Yahoo Finance](https://finance.yahoo.com/)

### Requirements
```bash
pip install yfinance pandas numpy google-generativeai
```

---

## 🐛 Troubleshooting

### Common Issues

**"No data found for symbol"**
- Check ticker symbol is correct
- Try a different time period
- Verify symbol exists on Yahoo Finance

**"AI scoring failed"**
- Check Gemini API key is valid
- Verify internet connection
- API may have rate limits

**"Module not found"**
- Install required packages: `pip install -r requirements.txt`
- Ensure Python 3.7+ is installed

**"Permission denied" when saving**
- Check write permissions in directory
- Run with appropriate user permissions

---

## 📝 Signal Category Summary

| Category | Signals | Focus |
|----------|---------|-------|
| MA_CROSS | 8 | Moving average crossovers |
| MA_TREND | 4 | MA alignment and positioning |
| MA_COMPRESSION | 1 | MA convergence |
| MA_SLOPE | 2 | MA direction changes |
| MA_DISTANCE | 2 | Price distance from MAs |
| RSI | 4 | RSI levels |
| RSI_MOMENTUM | 2 | RSI velocity |
| RSI_DIVERGENCE | 2 | Price/RSI divergence |
| RSI_CROSS | 2 | RSI 50-line crosses |
| MACD | 4 | MACD line crosses |
| MACD_MOMENTUM | 2 | MACD histogram |
| BOLLINGER | 2 | BB touches |
| BB_SQUEEZE | 1 | BB compression |
| BB_WALK | 2 | BB riding |
| BB_EXTREME | 2 | BB extensions |
| VOLUME | 4 | Volume spikes |
| VOLUME_DIVERGENCE | 2 | Volume/price divergence |
| VOLUME_ACCUMULATION | 1 | Volume trends |
| VOLUME_CLIMAX | 2 | Extreme volume |
| PRICE_ACTION | 4 | Large moves |
| PRICE_PATTERN | 4 | Price formations |
| RANGE | 2 | 52-week extremes |
| GAP | 2 | Opening gaps |
| CANDLESTICK | 5 | Candle patterns |
| PIVOT | 2 | Pivot breaks |
| FIBONACCI | 2 | Fib levels |
| STRUCTURE_BREAK | 2 | Support/resistance breaks |
| STOCHASTIC | 4 | Stochastic signals |
| WILLIAMS_R | 2 | Williams %R |
| CCI | 2 | Commodity Channel Index |
| MFI | 2 | Money Flow Index |
| TREND | 2 | ADX strength |
| ADX | 4 | ADX with DI |
| OBV | 1 | On-Balance Volume |
| ICHIMOKU | 3 | Ichimoku cloud |
| ROC | 2 | Rate of Change |
| VWAP | 2 | VWAP crosses |
| ATR | 1 | Stop levels |
| EXHAUSTION | 2 | Climax signals |
| TIMEFRAME_ALIGNMENT | 2 | Multi-TF confirmation |
| MULTI_INDICATOR | 2 | Cross-indicator |
| CONSOLIDATION | 1 | Range compression |
| MOMENTUM_DIVERGENCE | 1 | Price/volume divergence |
| MOMENTUM_ACCELERATION | 2 | Momentum changes |
| VOLATILITY | 2 | Volatility levels |
| VOLATILITY_SQUEEZE | 1 | Vol compression |

**Total: 150+ signals across 40+ categories**

---

## 🎯 Quick Reference Card

### Top 10 Most Reliable Signals
1. **Golden Cross / Death Cross** - Major trend changes
2. **MA Alignment** - Trend confirmation
3. **Volume Breakout/Selloff** - Institutional activity
4. **MACD Histogram Expansion** - Momentum acceleration
5. **Bullish/Bearish Engulfing** - Reversal patterns
6. **Full Timeframe Alignment** - All TFs agree
7. **RSI Divergence** - Hidden momentum
8. **BB Walk** - Strong trending
9. **Structure Break** - Support/resistance violations
10. **Multi-Indicator Confirmation** - Cross-validation

### Top 5 Early Warning Signals
1. **Volume Divergence** - Trend weakening
2. **MA Compression** - Breakout pending
3. **BB Squeeze** - Volatility expansion coming
4. **RSI Bearish Divergence** - Top forming
5. **Inside Bar** - Consolidation before move

### Top 5 Confirmation Signals
1. **Volume Accumulation** - Confirms uptrend
2. **OBV Confirmation** - Volume validates price
3. **DI Cross** - Directional confirmation
4. **Above VWAP** - Institutional support
5. **Stochastic Bull Cross** - Momentum confirmed

---

## 📖 Glossary

**ADX** - Average Directional Index: Measures trend strength (0-100)

**ATR** - Average True Range: Measures volatility in price units

**Bollinger Bands** - Volatility bands around moving average

**CCI** - Commodity Channel Index: Momentum oscillator

**Divergence** - When price and indicator move in opposite directions

**DI** - Directional Indicator: +DI (up) and -DI (down) components of ADX

**EMA** - Exponential Moving Average: Gives more weight to recent prices

**Fibonacci** - Retracement levels based on golden ratio (61.8%, 50%, 38.2%)

**Ichimoku Cloud** - Japanese indicator showing support/resistance zones

**MACD** - Moving Average Convergence Divergence: Trend-following momentum

**MFI** - Money Flow Index: Volume-weighted RSI

**OBV** - On-Balance Volume: Cumulative volume indicator

**Pivot Points** - Support/resistance levels calculated from previous period

**ROC** - Rate of Change: Momentum indicator showing % price change

**RSI** - Relative Strength Index: Momentum oscillator (0-100)

**SMA** - Simple Moving Average: Average price over period

**Stochastic** - Momentum indicator comparing close to high-low range

**VWAP** - Volume-Weighted Average Price: Institutional benchmark

**Williams %R** - Momentum indicator similar to Stochastic

---

## 🚀 Getting Started Checklist

- [ ] Install required packages
- [ ] Get Gemini API key (optional but recommended)
- [ ] Set environment variable for API key
- [ ] Choose symbol and time period
- [ ] Run first analysis
- [ ] Review top 10 signals
- [ ] Check category breakdown
- [ ] Export results to CSV
- [ ] Analyze technical data
- [ ] Implement risk management
- [ ] Start paper trading signals
- [ ] Track performance
- [ ] Refine strategy based on results

---

## 📞 Support & Feedback

For issues, questions, or feature requests, please review the code documentation and examples provided in the main Python file.

---

**Version:** 1.0.0  
**Last Updated:** 2025-01-06  
**Signals:** 150+  
**Indicators:** 80+  
**Categories:** 40+

---

**Happy Trading! 📈**
