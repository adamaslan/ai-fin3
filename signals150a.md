# ============================================================================
# COMPREHENSIVE SIGNAL CALCULATIONS REFERENCE (150+ Signals)
# ============================================================================
# This document lists all signals, their calculations, and alternative methods

"""
SIGNAL CALCULATIONS ORGANIZED BY CATEGORY
Each signal includes:
1. Name & Description
2. Current Calculation
3. Alternative/Industry Standard Methods (in comments)
4. Detection Logic in Code
"""

# ============================================================================
# 1. MOVING AVERAGE SIGNALS (MA_CROSS, MA_TREND, MA_COMPRESSION, etc.)
# ============================================================================

# SIGNAL #1: GOLDEN CROSS
# Calculation: SMA_50 > SMA_200 (today) AND SMA_50 <= SMA_200 (yesterday)
# Code: prev['SMA_50'] <= prev['SMA_200'] and current['SMA_50'] > current['SMA_200']
# Alternative: Use EMA_50 > EMA_200 (faster, more responsive to price)
# Alternative: Require both price above 200 SMA for stronger confirmation
# Category: MA_CROSS | Strength: STRONG BULLISH

# SIGNAL #2: DEATH CROSS
# Calculation: SMA_50 < SMA_200 (today) AND SMA_50 >= SMA_200 (yesterday)
# Code: prev['SMA_50'] >= prev['SMA_200'] and current['SMA_50'] < current['SMA_200']
# Alternative: EMA_50 < EMA_200 (faster moving average crossover)
# Alternative: Add volume confirmation requirement
# Category: MA_CROSS | Strength: STRONG BEARISH

# SIGNAL #3: PRICE ABOVE 10 MA
# Calculation: Close <= SMA_10 (yesterday) AND Close > SMA_10 (today)
# Code: prev['Close'] <= prev['SMA_10'] and current['Close'] > current['SMA_10']
# Alternative: Use EMA_10 instead (more responsive to recent price)
# Alternative: Price crosses above SMA_20 for less noise
# Category: MA_CROSS | Strength: BULLISH

# SIGNAL #4: PRICE BELOW 10 MA
# Calculation: Close >= SMA_10 (yesterday) AND Close < SMA_10 (today)
# Code: prev['Close'] >= prev['SMA_10'] and current['Close'] < current['SMA_10']
# Alternative: EMA_10 for faster response
# Alternative: Use SMA_20 for stronger signal
# Category: MA_CROSS | Strength: BEARISH

# SIGNAL #5: 10/20 EMA BULL CROSS
# Calculation: EMA_10 > EMA_20 (today) AND EMA_10 <= EMA_20 (yesterday)
# Code: prev['EMA_10'] <= prev['EMA_20'] and current['EMA_10'] > current['EMA_20']
# Alternative: Use SMA for slower, more confirmed moves
# Alternative: Require both above SMA_50 for confluence
# Category: MA_CROSS | Strength: BULLISH

# SIGNAL #6: 10/20 EMA BEAR CROSS
# Calculation: EMA_10 < EMA_20 (today) AND EMA_10 >= EMA_20 (yesterday)
# Code: prev['EMA_10'] >= prev['EMA_20'] and current['EMA_10'] < current['EMA_20']
# Alternative: Use SMA crossover (more reliable, less whipsaw)
# Alternative: Require price below SMA_50
# Category: MA_CROSS | Strength: BEARISH

# SIGNAL #7: MA ALIGNMENT BULLISH
# Calculation: SMA_10 > SMA_20 > SMA_50 (all in order)
# Code: current['SMA_10'] > current['SMA_20'] > current['SMA_50']
# Alternative: Add SMA_100 > SMA_200 for additional confirmation
# Alternative: Use EMA versions for faster detection
# Category: MA_TREND | Strength: STRONG BULLISH

# SIGNAL #8: MA ALIGNMENT BEARISH
# Calculation: SMA_10 < SMA_20 < SMA_50 (all in order)
# Code: current['SMA_10'] < current['SMA_20'] < current['SMA_50']
# Alternative: Add EMA versions for confluence
# Alternative: Require price below all three MAs
# Category: MA_TREND | Strength: STRONG BEARISH

# SIGNAL #9: ABOVE 200 SMA
# Calculation: Close > SMA_200 (requires len(df) > 200)
# Code: current['Close'] > current['SMA_200']
# Alternative: Used as baseline for long-term bullish bias
# Alternative: More reliable when combined with other bullish signals
# Category: MA_TREND | Strength: BULLISH

# SIGNAL #10: BELOW 200 SMA
# Calculation: Close < SMA_200 (requires len(df) > 200)
# Code: current['Close'] < current['SMA_200']
# Alternative: Used as baseline for long-term bearish bias
# Alternative: Strongest signal when combined with other bearish indicators
# Category: MA_TREND | Strength: BEARISH

# SIGNAL #11: MA COMPRESSION
# Calculation: (SMA_50 - SMA_10) / SMA_50 * 100 < 2% (abs value)
# Code: ma_range = (current['SMA_50'] - current['SMA_10']) / current['SMA_50'] * 100
#       if abs(ma_range) < 2:
# Alternative: Calculate BBW (Bollinger Band Width) / ratio for convergence
# Alternative: Use standard deviation of MAs to detect compression
# Alternative: Set threshold at 1.5% for tighter compression
# Category: MA_COMPRESSION | Strength: NEUTRAL

# SIGNAL #12: MA SLOPE REVERSAL UP
# Calculation: SMA_20_Slope > 0 (today) AND SMA_20_Slope <= 0 (yesterday)
# Slope calculation: df[f'SMA_20_Slope'] = df[f'SMA_20'].diff(5)
# Code: current.get('SMA_20_Slope', 0) > 0 and prev.get('SMA_20_Slope', 0) <= 0
# Alternative: Use slope over 10 periods for more dramatic turns
# Alternative: Calculate slope as rise/run: (MA[today] - MA[5 days ago]) / 5
# Alternative: Use linear regression slope for smoother trend detection
# Category: MA_SLOPE | Strength: BULLISH

# SIGNAL #13: MA SLOPE REVERSAL DOWN
# Calculation: SMA_20_Slope < 0 (today) AND SMA_20_Slope >= 0 (yesterday)
# Slope calculation: df[f'SMA_20_Slope'] = df[f'SMA_20'].diff(5)
# Code: current.get('SMA_20_Slope', 0) < 0 and prev.get('SMA_20_Slope', 0) >= 0
# Alternative: Use 10-period slope for more significant reversal
# Alternative: Linear regression (polyfit) for more statistical rigor
# Category: MA_SLOPE | Strength: BEARISH

# SIGNAL #14: OVEREXTENDED ABOVE 20MA
# Calculation: (Close - SMA_20) / SMA_20 * 100 > 10%
# Code: current.get('Dist_SMA_20', 0) > 10
# Alternative: Use 15% threshold for less sensitivity
# Alternative: Scale threshold by ATR: Distance > 2 * ATR
# Alternative: Use Bollinger Bands instead (more standard)
# Category: MA_DISTANCE | Strength: BEARISH

# SIGNAL #15: OVEREXTENDED BELOW 20MA
# Calculation: (Close - SMA_20) / SMA_20 * 100 < -10%
# Code: current.get('Dist_SMA_20', 0) < -10
# Alternative: Use -15% for less sensitivity
# Alternative: Scale by volatility: Distance < -2 * ATR
# Alternative: Use Bollinger Band position instead
# Category: MA_DISTANCE | Strength: BULLISH

# ============================================================================
# 2. RSI SIGNALS (RSI, RSI_MOMENTUM, RSI_DIVERGENCE, RSI_CROSS)
# ============================================================================

# SIGNAL #16: RSI OVERSOLD
# Calculation: RSI < 30
# RSI Formula: RSI = 100 - (100 / (1 + RS)) where RS = AvgGain / AvgLoss (14-period)
# Code: if current['RSI'] < 30:
# Alternative: Use RSI < 35 for earlier detection
# Alternative: Wilder's smoothing method:
#   gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#   loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
# Alternative: EMA-based RSI for different weighting
# Category: RSI | Strength: BULLISH

# SIGNAL #17: RSI OVERBOUGHT
# Calculation: RSI > 70
# Code: if current['RSI'] > 70:
# Alternative: Use RSI > 65 for earlier detection
# Alternative: Adjust threshold based on volatility regime
# Category: RSI | Strength: BEARISH

# SIGNAL #18: RSI EXTREME OVERSOLD
# Calculation: RSI < 20
# Code: if current['RSI'] < 20:
# Alternative: RSI < 25 for slightly less extreme
# Category: RSI | Strength: STRONG BULLISH

# SIGNAL #19: RSI EXTREME OVERBOUGHT
# Calculation: RSI > 80
# Code: if current['RSI'] > 80:
# Alternative: RSI > 75 for less extreme
# Category: RSI | Strength: STRONG BEARISH

# SIGNAL #20: RSI BULLISH DIVERGENCE
# Calculation: Price makes lower low but RSI makes higher high (20 periods)
# Code: if current['Close'] < df['Close'].iloc[-20] and 
#          current['RSI'] > df['RSI'].iloc[-20]:
# Alternative: Use 10 periods for faster detection
# Alternative: Use hidden bullish divergence: Higher lows on price, lower lows on RSI
# Alternative: Use regression to detect divergence more statistically
# Category: RSI_DIVERGENCE | Strength: BULLISH

# SIGNAL #21: RSI BEARISH DIVERGENCE
# Calculation: Price makes higher high but RSI makes lower high (20 periods)
# Code: if current['Close'] > df['Close'].iloc[-20] and 
#          current['RSI'] < df['RSI'].iloc[-20]:
# Alternative: Use 10-period lookback
# Alternative: Hidden bearish divergence: Lower highs on price, higher highs on RSI
# Category: RSI_DIVERGENCE | Strength: BEARISH

# SIGNAL #22: RSI MOMENTUM SURGE
# Calculation: RSI_Change = RSI[today] - RSI[5 days ago] > 10 AND RSI < 50
# Code: rsi_momentum = current['RSI'] - df['RSI'].iloc[-5]
#       if rsi_momentum > 10 and current['RSI'] < 50:
# Alternative: Use RSI_Change > 15 for stronger signal
# Alternative: Calculate rate of change: (RSI[today] / RSI[5d ago]) > 1.15
# Category: RSI_MOMENTUM | Strength: BULLISH

# SIGNAL #23: RSI MOMENTUM COLLAPSE
# Calculation: RSI_Change = RSI[today] - RSI[5 days ago] < -10 AND RSI > 50
# Code: elif rsi_momentum < -10 and current['RSI'] > 50:
# Alternative: Use threshold of -15 for stronger signal
# Alternative: Use ratio of RSI changes
# Category: RSI_MOMENTUM | Strength: BEARISH

# SIGNAL #24: RSI ABOVE 50
# Calculation: RSI crosses above 50 line
# Code: if prev['RSI'] <= 50 and current['RSI'] > 50:
# Alternative: RSI > 55 for slightly more bullish threshold
# Alternative: Use as confirmation signal rather than standalone
# Category: RSI_CROSS | Strength: BULLISH

# SIGNAL #25: RSI BELOW 50
# Calculation: RSI crosses below 50 line
# Code: if prev['RSI'] >= 50 and current['RSI'] < 50:
# Alternative: RSI < 45 for stronger bearish signal
# Category: RSI_CROSS | Strength: BEARISH

# ============================================================================
# 3. MACD SIGNALS (MACD, MACD_MOMENTUM)
# ============================================================================

# SIGNAL #26: MACD BULL CROSS
# Calculation: MACD > MACD_Signal (today) AND MACD <= MACD_Signal (yesterday)
# MACD Formula: 
#   EMA_12 = df['Close'].ewm(span=12, adjust=False).mean()
#   EMA_26 = df['Close'].ewm(span=26, adjust=False).mean()
#   MACD = EMA_12 - EMA_26
#   MACD_Signal = MACD.ewm(span=9, adjust=False).mean()
# Code: prev['MACD'] <= prev['MACD_Signal'] and current['MACD'] > current['MACD_Signal']
# Alternative: Wait for MACD histogram to turn positive for confirmation
# Alternative: Use standard MACD settings (12, 26, 9) - already using these
# Alternative: Faster MACD (5, 35, 5) for quicker signals
# Category: MACD | Strength: BULLISH

# SIGNAL #27: MACD BEAR CROSS
# Calculation: MACD < MACD_Signal (today) AND MACD >= MACD_Signal (yesterday)
# Code: prev['MACD'] >= prev['MACD_Signal'] and current['MACD'] < current['MACD_Signal']
# Alternative: Wait for histogram to turn negative for confirmation
# Alternative: Use fast MACD settings (5, 35, 5)
# Category: MACD | Strength: BEARISH

# SIGNAL #28: MACD ABOVE ZERO
# Calculation: MACD crosses from negative to positive
# Code: prev['MACD'] <= 0 and current['MACD'] > 0
# Alternative: Require signal line also positive for confluence
# Alternative: Use as confirmation of bullish trend
# Category: MACD | Strength: BULLISH

# SIGNAL #29: MACD BELOW ZERO
# Calculation: MACD crosses from positive to negative
# Code: prev['MACD'] >= 0 and current['MACD'] < 0
# Alternative: Require signal line negative for confluence
# Alternative: Use as bearish trend confirmation
# Category: MACD | Strength: BEARISH

# SIGNAL #30: MACD HISTOGRAM EXPANSION (Bullish)
# Calculation: |MACD_Hist| > |MACD_Hist[yesterday]| > |MACD_Hist[2 days ago]|
#              AND MACD_Hist > 0
# MACD_Hist = MACD - MACD_Signal
# Code: hist_expanding = abs(current['MACD_Hist']) > abs(prev['MACD_Hist']) > 
#                        abs(prev2['MACD_Hist'])
#       if hist_expanding and current['MACD_Hist'] > 0:
# Alternative: Use percentage change instead of absolute values
# Alternative: Require 3+ consecutive days of expansion
# Category: MACD_MOMENTUM | Strength: STRONG BULLISH

# SIGNAL #31: MACD HISTOGRAM EXPANSION (Bearish)
# Calculation: Same as above but with MACD_Hist < 0
# Code: elif hist_expanding and current['MACD_Hist'] < 0:
# Alternative: Use ratio of histogram sizes for stronger signal
# Category: MACD_MOMENTUM | Strength: STRONG BEARISH

# ============================================================================
# 4. BOLLINGER BANDS SIGNALS (BOLLINGER, BB_SQUEEZE, BB_WALK, BB_EXTREME)
# ============================================================================

# SIGNAL #32: BOLLINGER BAND SQUEEZE
# Calculation: BB_Width < (Average BB_Width over 50 periods * 0.7)
# BB_Width = Upper_Band - Lower_Band
# BB Calculation:
#   Middle = SMA_20
#   Upper = Middle + (2 * StdDev_20)
#   Lower = Middle - (2 * StdDev_20)
# Code: bb_width_avg = df['BB_Width'].tail(50).mean()
#       if current['BB_Width'] < bb_width_avg * 0.7:
# Alternative: Use Bollinger Band Width indicator (BBW) directly
# Alternative: Use keltner channel squeeze for comparison
# Alternative: Threshold at 0.5 (50% of average) for tighter squeeze
# Category: BB_SQUEEZE | Strength: NEUTRAL

# SIGNAL #33: AT LOWER BOLLINGER BAND
# Calculation: Close <= Lower_Band * 1.01 (1% above lower band)
# Code: if current['Close'] <= current['BB_Lower'] * 1.01:
# Alternative: Use Close <= Lower_Band exactly (no buffer)
# Alternative: Use BB position ratio: (Close - Lower) / (Upper - Lower) < 0.1
# Category: BOLLINGER | Strength: BULLISH

# SIGNAL #34: AT UPPER BOLLINGER BAND
# Calculation: Close >= Upper_Band * 0.99 (1% below upper band)
# Code: if current['Close'] >= current['BB_Upper'] * 0.99:
# Alternative: Close >= Upper_Band exactly
# Alternative: Use position ratio > 0.9
# Category: BOLLINGER | Strength: BEARISH

# SIGNAL #35: BB WALK UPPER
# Calculation: Price within 2% of upper band for 3 consecutive days
# Code: bb_walk_bull = all(df['Close'].iloc[-i] >= df['BB_Upper'].iloc[-i] * 0.98 
#                          for i in range(1, 4))
# Alternative: Use 5 days for stronger confirmation
# Alternative: Price touches upper band and bounces (reversal signal)
# Alternative: Linear regression of price vs upper band
# Category: BB_WALK | Strength: STRONG BULLISH

# SIGNAL #36: BB WALK LOWER
# Calculation: Price within 2% of lower band for 3 consecutive days
# Code: bb_walk_bear = all(df['Close'].iloc[-i] <= df['BB_Lower'].iloc[-i] * 1.02 
#                          for i in range(1, 4))
# Alternative: Use 5 days for stronger signal
# Alternative: Detect bounces off lower band
# Category: BB_WALK | Strength: STRONG BEARISH

# SIGNAL #37: ABOVE UPPER BOLLINGER BAND (Extreme)
# Calculation: Close > Upper_Band (price extends beyond band)
# BB Position = (Close - Lower) / (Upper - Lower)
# Code: if current['BB_Position'] > 1.1:
# Alternative: Use 1.05 for less extreme threshold
# Alternative: Measure distance in multiples of BB width
# Category: BB_EXTREME | Strength: EXTREME BULLISH

# SIGNAL #38: BELOW LOWER BOLLINGER BAND (Extreme)
# Calculation: Close < Lower_Band (price extends beyond band)
# Code: if current['BB_Position'] < -0.1:
# Alternative: Use -0.05 threshold
# Category: BB_EXTREME | Strength: EXTREME BEARISH

# ============================================================================
# 5. VOLUME SIGNALS (VOLUME, VOLUME_DIVERGENCE, VOLUME_ACCUMULATION, etc.)
# ============================================================================

# SIGNAL #39: VOLUME SPIKE 2X
# Calculation: Volume > (20-day Avg Volume * 2)
# Volume_MA_20 = df['Volume'].rolling(window=20).mean()
# Code: if current['Volume'] > current['Volume_MA_20'] * 2:
# Alternative: Use 1.5x for more frequent signals
# Alternative: Use standard deviations: Vol > (Mean + 2*StdDev)
# Category: VOLUME | Strength: SIGNIFICANT

# SIGNAL #40: EXTREME VOLUME 3X
# Calculation: Volume > (20-day Avg Volume * 3)
# Code: if current['Volume'] > current['Volume_MA_20'] * 3:
# Alternative: Use 2.5x for slightly less extreme
# Alternative: Use percentile ranking (e.g., > 95th percentile)
# Category: VOLUME | Strength: VERY SIGNIFICANT

# SIGNAL #41: VOLUME BREAKOUT
# Calculation: Price_Change > 2% AND Volume > (20-day Avg Volume * 1.5)
# Code: if current['Price_Change'] > 2 and 
#          current['Volume'] > current['Volume_MA_20'] * 1.5:
# Alternative: Use 3% price move for more reliable signal
# Alternative: Require volume > 50-day average for stronger confirmation
# Category: VOLUME | Strength: STRONG BULLISH

# SIGNAL #42: VOLUME SELLOFF
# Calculation: Price_Change < -2% AND Volume > (20-day Avg Volume * 1.5)
# Code: if current['Price_Change'] < -2 and 
#          current['Volume'] > current['Volume_MA_20'] * 1.5:
# Alternative: Use 3% down move for more reliability
# Alternative: Compare to 50-day volume average
# Category: VOLUME | Strength: STRONG BEARISH

# SIGNAL #43: VOLUME DIVERGENCE BEARISH
# Calculation: Price up 10+ days, but Volume down 10 days
# Code: price_trend_up = current['Close'] > df['Close'].iloc[-10]
#       volume_trend_down = current['Volume'] < df['Volume'].iloc[-10]
# Alternative: Use longer timeframe (20 days) for more significance
# Alternative: Calculate ROC of price vs ROC of volume
# Category: VOLUME_DIVERGENCE | Strength: BEARISH

# SIGNAL #44: VOLUME DIVERGENCE BULLISH
# Calculation: Price down 10 days, but Volume up 10 days
# Code: elif not price_trend_up and not volume_trend_down:
# Alternative: Use 20-day lookback
# Alternative: Use correlation of price vs volume ROC
# Category: VOLUME_DIVERGENCE | Strength: BULLISH

# SIGNAL #45: VOLUME ACCUMULATION
# Calculation: Volume increases for 3 consecutive days AND Price increases
# Code: volume_increasing = all(df['Volume'].iloc[-i] > df['Volume'].iloc[-i-1] 
#                                for i in range(1, 4))
#       if volume_increasing and current['Close'] > prev['Close']:
# Alternative: Use OBV (On-Balance Volume) for more sophisticated accumulation
# Alternative: Require 5+ consecutive days of volume increase
# Category: VOLUME_ACCUMULATION | Strength: STRONG BULLISH

# SIGNAL #46: VOLUME CLIMAX (Buying)
# Calculation: Volume > (20-day Avg Volume * 4) AND Close > Open
# Code: if current['Volume'] > current['Volume_MA_20'] * 4:
#           if current['Close'] > current['Open']:
# Alternative: Use 3x for less extreme threshold
# Alternative: Measure distance moved: High - Low relative to volume
# Category: VOLUME_CLIMAX | Strength: EXTREME

# SIGNAL #47: VOLUME CLIMAX (Selling)
# Calculation: Volume > (20-day Avg Volume * 4) AND Close < Open
# Code: else:  # current['Close'] < current['Open']
# Alternative: Use 3x threshold
# Category: VOLUME_CLIMAX | Strength: EXTREME

# ============================================================================
# 6. PRICE ACTION SIGNALS (PRICE_ACTION, GAP, CANDLESTICK)
# ============================================================================

# SIGNAL #48: LARGE GAIN
# Calculation: Daily % Change > 5%
# Price_Change = (Close - Close[yesterday]) / Close[yesterday] * 100
# Code: if current['Price_Change'] > 5:
# Alternative: Use 3% threshold for more frequent signals
# Alternative: Scale by volatility: Change > (Volatility * 1.5)
# Category: PRICE_ACTION | Strength: STRONG BULLISH

# SIGNAL #49: LARGE LOSS
# Calculation: Daily % Change < -5%
# Code: if current['Price_Change'] < -5:
# Alternative: Use -3% for more sensitivity
# Alternative: Scale by volatility regime
# Category: PRICE_ACTION | Strength: STRONG BEARISH

# SIGNAL #50: 52-WEEK HIGH
# Calculation: Close >= 52-week High * 99.9%
# High_52w = df['High'].rolling(window=252).max()
# Code: if current['Close'] >= current['High_52w'] * 0.999:
# Alternative: Use exact 52-week high (== instead of >=)
# Alternative: Use 99.5% for slightly less stringent
# Category: RANGE | Strength: STRONG BULLISH

# SIGNAL #51: 52-WEEK LOW
# Calculation: Close <= 52-week Low * 100.1%
# Low_52w = df['Low'].rolling(window=252).min()
# Code: if current['Close'] <= current['Low_52w'] * 1.001:
# Alternative: Exact low (== instead of <=)
# Alternative: Use 100.5% threshold
# Category: RANGE | Strength: STRONG BEARISH

# SIGNAL #52: GAP UP
# Calculation: (Open - Previous Close) / Previous Close * 100 > 2%
# Code: gap_up = (current['Open'] - prev['Close']) / prev['Close'] * 100
#       if gap_up > 2:
# Alternative: Use 1.5% for more frequent gap detection
# Alternative: Use ATR as basis: Gap > 0.5 * ATR
# Category: GAP | Strength: BULLISH

# SIGNAL #53: GAP DOWN
# Calculation: (Open - Previous Close) / Previous Close * 100 < -2%
# Code: elif gap_up < -2:
# Alternative: Use -1.5% threshold
# Alternative: Scale by ATR
# Category: GAP | Strength: BEARISH

# SIGNAL #54: BULLISH ENGULFING
# Calculation: 
#   Close > Open (bullish candle today)
#   Close[yesterday] < Open[yesterday] (bearish candle yesterday)
#   Open[today] <= Close[yesterday] (gap or overlap)
#   Close[today] >= Open[yesterday] (fully engulfing)
# Code: if current['Close'] > current['Open'] and 
#          prev['Close'] < prev['Open'] and
#          current['Open'] <= prev['Close'] and 
#          current['Close'] >= prev['Open']:
# Alternative: Require current body > previous body size
# Alternative: Use with other bullish indicators for confirmation
# Category: CANDLESTICK | Strength: STRONG BULLISH

# SIGNAL #55: BEARISH ENGULFING
# Calculation: Reverse of bullish engulfing
# Code: if current['Close'] < current['Open'] and 
#          prev['Close'] > prev['Open'] and
#          current['Open'] >= prev['Close'] and 
#          current['Close'] <= prev['Open']:
# Alternative: Require current body > previous body
# Category: CANDLESTICK | Strength: STRONG BEARISH

# SIGNAL #56: DOJI CANDLE
# Calculation: Candle Body / Candle Range < 10% (indecision)
# Body_Size = abs(Close - Open)
# Candle_Range = High - Low
# Code: body_size = abs(current['Close'] - current['Open'])
#       candle_range = current['High'] - current['Low']
#       if candle_range > 0 and body_size / candle_range < 0.1:
# Alternative: Use 15% threshold for more common dojis
# Alternative: Require upper and lower wicks of similar length
# Category: CANDLESTICK | Strength: NEUTRAL

# SIGNAL #57: HAMMER PATTERN
# Calculation: 
#   Lower_Wick > 2 * Body_Size (long lower shadow)
#   Upper_Wick < Body_Size (short upper shadow)
#   Usually occurs after downtrend
# Code: lower_wick = min(current['Open'], current['Close']) - current['Low']
#       upper_wick = current['High'] - max(current['Open'], current['Close'])
#       if lower_wick > 2 * body_size and upper_wick < body_size:
# Alternative: Require lower wick > 1.5x body
# Alternative: Require hammer at support level
# Category: CANDLESTICK | Strength: BULLISH

# SIGNAL #58: SHOOTING STAR
# Calculation:
#   Upper_Wick > 2 * Body_Size (long upper shadow)
#   Lower_Wick < Body_Size (short lower shadow)
# Code: if upper_wick > 2 * body_size and lower_wick < body_size:
# Alternative: Require upper wick > 1.5x body
# Alternative: Require at resistance level
# Category: CANDLESTICK | Strength: BEARISH

# SIGNAL #59: INSIDE BAR
# Calculation: Today's High < Yesterday's High AND Today's Low > Yesterday's Low
# Code: if current['High'] < prev['High'] and current['Low'] > prev['Low']:
# Alternative: Require 3+ consecutive inside bars for stronger pattern
# Alternative: Use inside bar breakout confirmation
# Category: PRICE_PATTERN | Strength: NEUTRAL

# SIGNAL #60: OUTSIDE BAR
# Calculation: Today's High > Yesterday's High AND Today's Low < Yesterday's Low
# Code: if current['High'] > prev['High'] and current['Low'] < prev['Low']:
# Alternative: Require higher volume for expansion
# Alternative: Use as reversal or continuation signal
# Category: PRICE_PATTERN | Strength: SIGNIFICANT

# ============================================================================
# 7. TREND & STRUCTURE SIGNALS (TREND, STRUCTURE_BREAK, RANGE)
# ============================================================================

# SIGNAL #61: STRONG UPTREND
# Calculation: ADX > 25 AND Close > SMA_50
# ADX Formula: (complex calculation involving directional indicators)
#   TR = max(High - Low, |High - Close[yesterday]|, |Low - Close[yesterday]|)
#   +DM = High - High[yesterday] (if > 0, else 0)
#   -DM = Low[yesterday] - Low (if > 0, else 0)
#   +DI = 100 * (+DM / ATR_14)
#   -DI = 100 * (-DM / ATR_14)
#   DX = 100 * |+DI - -DI| / (+DI + -DI)
#   ADX = EMA(DX, 14)
# Code: if current['ADX'] > 25:
#           trend = 'UP' if current['Close'] > current['SMA_50'] else 'DOWN'
# Alternative: Use ADX > 20 for earlier detection
# Alternative: Use +DI > -DI confirmation
# Category: TREND | Strength: TRENDING

# SIGNAL #62: STRONG DOWNTREND
# Calculation: ADX > 25 AND Close < SMA_50
# Code: (same as above with opposite condition)
# Alternative: ADX > 20
# Category: TREND | Strength: TRENDING

# SIGNAL #63: VERY STRONG TREND
# Calculation: ADX > 40 (extremely strong trend)
# Code: if current['ADX'] > 40:
# Alternative: ADX > 35 for slightly lower threshold
# Category: TREND | Strength: EXTREME

# SIGNAL #64: RESISTANCE BREAKOUT
# Calculation: Close > Resistance_Level (20-day high) today, was <= yesterday
# Resistance_Level = High.rolling(20).max()
# Code: resistance_level = df['High'].iloc[-20:].max()
#       if current['Close'] > resistance_level and prev['Close'] <= resistance_level:
# Alternative: Use 50-day high for stronger resistance
# Alternative: Require volume confirmation (> average volume)
# Category: STRUCTURE_BREAK | Strength: STRONG BULLISH

# SIGNAL #65: SUPPORT BREAKDOWN
# Calculation: Close < Support_Level (20-day low) today, was >= yesterday
# Support_Level = Low.rolling(20).min()
# Code: support_level = df['Low'].iloc[-20:].max()  # NOTE: This seems like bug in code
#       if current['Close'] < support_level and prev['Close'] >= support_level:
# Alternative: Use 50-day low
# Alternative: Require high volume
# Category: STRUCTURE_BREAK | Strength: STRONG BEARISH

# SIGNAL #66: HIGHER HIGHS & LOWS
# Calculation: All recent highs ascending AND all recent lows ascending (5 periods)
# Code: higher_highs = all(recent_highs[i] > recent_highs[i+1] for i in range(len(recent_highs)-1))
#       higher_lows = all(recent_lows[i] > recent_lows[i+1] for i in range(len(recent_lows)-1))
#       if higher_highs and higher_lows:
# Alternative: Use 3 periods for faster detection
# Alternative: Use regression line slope to detect trend
# Category: PRICE_PATTERN | Strength: STRONG BULLISH

# SIGNAL #67: LOWER HIGHS & LOWS
# Calculation: All recent highs descending AND all recent lows descending (5 periods)
# Code: lower_highs = all(recent_highs[i] < recent_highs[i+1] ...)
#       lower_lows = all(recent_lows[i] < recent_lows[i+1] ...)
# Alternative: Use 3 periods
# Alternative: Regression line approach
# Category: PRICE_PATTERN | Strength: STRONG BEARISH

# ============================================================================
# 8. OSCILLATOR SIGNALS (STOCHASTIC, WILLIAMS_R, CCI, MFI, ADX, OBV)
# ============================================================================

# SIGNAL #68: STOCHASTIC OVERSOLD
# Calculation: Stoch_K < 20 today, >= 20 yesterday
# Stochastic Formula:
#   K = 100 * (Close - Low[14]) / (High[14] - Low[14])
#   D = SMA_3(K)
# Code: if current['Stoch_K'] < 20 and prev['Stoch_K'] >= 20:
# Alternative: Use < 25 for more sensitive signal
# Alternative: Use K < 20 AND D < 20 for stronger confirmation
# Category: STOCHASTIC | Strength: BULLISH

# SIGNAL #69: STOCHASTIC OVERBOUGHT
# Calculation: Stoch_K > 80 today, <= 80 yesterday
# Code: if current['Stoch_K'] > 80 and prev['Stoch_K'] <= 80:
# Alternative: Use > 75 for more frequent signals
# Alternative: K > 80 AND D > 80 for stronger signal
# Category: STOCHASTIC | Strength: BEARISH

# SIGNAL #70: STOCHASTIC BULL CROSS
# Calculation: K_Line crosses above D_Line
# Code: if prev['Stoch_K'] <= prev['Stoch_D'] and 
#          current['Stoch_K'] > current['Stoch_D']:
# Alternative: Require crossing in oversold zone (K < 30) for stronger signal
# Alternative: Use fast stochastic (no D smoothing)
# Category: STOCHASTIC | Strength: BULLISH

# SIGNAL #71: STOCHASTIC BEAR CROSS
# Calculation: K_Line crosses below D_Line
# Code: if prev['Stoch_K'] >= prev['Stoch_D'] and 
#          current['Stoch_K'] < current['Stoch_D']:
# Alternative: Require in overbought zone (K > 70)
# Category: STOCHASTIC | Strength: BEARISH

# SIGNAL #72: WILLIAMS %R OVERSOLD
# Calculation: Williams %R < -80
# Williams %R = -100 * (High[14] - Close) / (High[14] - Low[14])
# Code: if current['Williams_R'] < -80:
# Alternative: Use < -75 for less extreme
# Alternative: Use < -90 for more extreme
# Category: WILLIAMS_R | Strength: BULLISH

# SIGNAL #73: WILLIAMS %R OVERBOUGHT
# Calculation: Williams %R > -20
# Code: if current['Williams_R'] > -20:
# Alternative: Use > -25
# Category: WILLIAMS_R | Strength: BEARISH

# SIGNAL #74: CCI OVERBOUGHT
# Calculation: CCI > 100
# CCI Formula: CCI = (TP - SMA(TP,20)) / (0.015 * StdDev(TP,20))
#   TP = (High + Low + Close) / 3
# Code: if current['CCI'] > 100:
# Alternative: Use > 75 for earlier detection
# Alternative: Use > 150 for more extreme signal
# Category: CCI | Strength: BEARISH

# SIGNAL #75: CCI OVERSOLD
# Calculation: CCI < -100
# Code: if current['CCI'] < -100:
# Alternative: Use < -75
# Alternative: Use < -150 for extreme
# Category: CCI | Strength: BULLISH

# SIGNAL #76: MFI OVERSOLD
# Calculation: MFI < 20
# MFI Formula:
#   TP = (High + Low + Close) / 3
#   MF = TP * Volume
#   MFI = 100 - (100 / (1 + (Positive MF / Negative MF)))
# Code: if current['MFI'] < 20:
# Alternative: Use < 25 for more sensitive
# Alternative: Use < 10 for more extreme
# Category: MFI | Strength: BULLISH

# SIGNAL #77: MFI OVERBOUGHT
# Calculation: MFI > 80
# Code: if current['MFI'] > 80:
# Alternative: Use > 75
# Alternative: Use > 90 for more extreme
# Category: MFI | Strength: BEARISH

# SIGNAL #78: STRONG BULL TREND (ADX with +DI)
# Calculation: ADX > 30 AND +DI > -DI
# Code: if current['ADX'] > 30:
#           if current['Plus_DI'] > current['Minus_DI']:
# Alternative: Use ADX > 25
# Category: ADX | Strength: STRONG BULLISH

# SIGNAL #79: STRONG BEAR TREND (ADX with -DI)
# Calculation: ADX > 30 AND -DI > +DI
# Code: else:  # -DI > +DI
# Alternative: ADX > 25
# Category: ADX | Strength: STRONG BEARISH

# SIGNAL #80: DI BULL CROSS
# Calculation: +DI crosses above -DI
# Code: if prev['Plus_DI'] <= prev['Minus_DI'] and 
#          current['Plus_DI'] > current['Minus_DI']:
# Alternative: Require ADX > 25 for confirmation
# Category: ADX | Strength: BULLISH

# SIGNAL #81: DI BEAR CROSS
# Calculation: -DI crosses above +DI
# Code: if prev['Plus_DI'] >= prev['Minus_DI'] and 
#          current['Plus_DI'] < current['Minus_DI']:
# Alternative: Require ADX > 25
# Category: ADX | Strength: BEARISH

# SIGNAL #82: OBV CONFIRMATION (Bullish)
# Calculation: OBV increased > 20% over 20 days AND Price increased
# OBV = Cumulative(Sign(Close Change) * Volume)
# Code: obv_trend = (current['OBV'] - df['OBV'].iloc[-20]) / 
#                   abs(df['OBV'].iloc[-20]) * 100
#       if obv_trend > 20 and current['Close'] > df['Close'].iloc[-20]:
# Alternative: Use 15% threshold for more sensitivity
# Alternative: Use OBV slope instead of percentage change
# Category: OBV | Strength: STRONG BULLISH

# SIGNAL #83: OBV CONFIRMATION (Bearish)
# Calculation: OBV decreased > 20% AND Price decreased
# Code: elif obv_trend < -20 and current['Close'] < df['Close'].iloc[-20]:
# Alternative: Use -15% threshold
# Category: OBV | Strength: STRONG BEARISH

# ============================================================================
# 9. ADVANCED INDICATORS (ICHIMOKU, ROC, VWAP, ATR, EXHAUSTION)
# ============================================================================

# SIGNAL #84: ABOVE ICHIMOKU CLOUD
# Calculation: Close > Senkou_A AND Close > Senkou_B (above cloud)
# Ichimoku Formula:
#   Tenkan = (High[9] + Low[9]) / 2
#   Kijun = (High[26] + Low[26]) / 2
#   Senkou_A = ((Tenkan + Kijun) / 2) shifted 26 periods
#   Senkou_B = ((High[52] + Low[52]) / 2) shifted 26 periods
# Code: if current['Close'] > current['Senkou_A'] and 
#          current['Close'] > current['Senkou_B']:
# Alternative: Require Senkou_A > Senkou_B (cloud in bullish formation)
# Category: ICHIMOKU | Strength: BULLISH

# SIGNAL #85: BELOW ICHIMOKU CLOUD
# Calculation: Close < Senkou_A AND Close < Senkou_B (below cloud)
# Code: if current['Close'] < current['Senkou_A'] and 
#          current['Close'] < current['Senkou_B']:
# Alternative: Require Senkou_A < Senkou_B
# Category: ICHIMOKU | Strength: BEARISH

# SIGNAL #86: TENKAN CROSS UP
# Calculation: Close crosses above Tenkan line
# Code: if prev['Close'] <= prev['Tenkan'] and 
#          current['Close'] > current['Tenkan']:
# Alternative: Require Tenkan > Kijun for confluence
# Category: ICHIMOKU | Strength: BULLISH

# SIGNAL #87: EXTREME ROC POSITIVE
# Calculation: 20-period Rate of Change > 15%
# ROC = ((Close - Close[20]) / Close[20]) * 100
# Code: if current['ROC_20'] > 15:
# Alternative: Use 10% for more frequent signals
# Alternative: Use 20% for more extreme moves
# Category: ROC | Strength: STRONG BULLISH

# SIGNAL #88: EXTREME ROC NEGATIVE
# Calculation: 20-period Rate of Change < -15%
# Code: if current['ROC_20'] < -15:
# Alternative: Use -10%
# Alternative: Use -20% for extreme
# Category: ROC | Strength: STRONG BEARISH

# SIGNAL #89: ABOVE VWAP
# Calculation: Close crosses above VWAP
# VWAP = Cumulative(Volume * TP) / Cumulative(Volume)
#   TP = (High + Low + Close) / 3
# Code: if current['Close'] > current['VWAP'] and 
#          prev['Close'] <= prev['VWAP']:
# Alternative: Use as level to stay above for bullish bias
# Alternative: Require higher volume than average for confirmation
# Category: VWAP | Strength: BULLISH

# SIGNAL #90: BELOW VWAP
# Calculation: Close crosses below VWAP
# Code: if current['Close'] < current['VWAP'] and 
#          prev['Close'] >= prev['VWAP']:
# Alternative: Use as support/resistance level
# Category: VWAP | Strength: BEARISH

# SIGNAL #91: ATR STOP LEVEL
# Calculation: Stop = Close - (ATR * 2) for bullish positions
# ATR = Average True Range = (14-period MA of True Range)
# TR = max(H - L, |H - C[yesterday]|, |L - C[yesterday]|)
# Code: stop_distance = current['ATR'] * 2
#       if current['Close'] > current['SMA_20']:
#           stop_level = current['Close'] - stop_distance
# Alternative: Use ATR * 1.5 for tighter stop
# Alternative: Use ATR * 2.5 for wider stop
# Category: ATR | Strength: NEUTRAL

# SIGNAL #92: BULLISH EXHAUSTION
# Calculation: RSI > 75 AND Volume > (20-day Avg * 2)
# Code: if current['RSI'] > 75 and 
#          current['Volume'] > current['Volume_MA_20'] * 2:
# Alternative: Use RSI > 80 for more extreme exhaustion
# Alternative: Combine with bearish divergence
# Category: EXHAUSTION | Strength: BEARISH

# SIGNAL #93: BEARISH EXHAUSTION
# Calculation: RSI < 25 AND Volume > (20-day Avg * 2)
# Code: if current['RSI'] < 25 and 
#          current['Volume'] > current['Volume_MA_20'] * 2:
# Alternative: Use RSI < 20
# Alternative: Combine with bullish divergence
# Category: EXHAUSTION | Strength: BULLISH

# ============================================================================
# 10. SUPPORT/RESISTANCE & FIBONACCI (PIVOT, FIBONACCI)
# ============================================================================

# SIGNAL #94: ABOVE R1 PIVOT
# Calculation: Close crosses above R1 resistance
# Pivot Formula:
#   Pivot = (H[yesterday] + L[yesterday] + C[yesterday]) / 3
#   R1 = 2 * Pivot - L[yesterday]
#   S1 = 2 * Pivot - H[yesterday]
#   R2 = Pivot + (H[yesterday] - L[yesterday])
#   S2 = Pivot - (H[yesterday] - L[yesterday])
# Code: if current['Close'] > current['R1'] and 
#          prev['Close'] <= prev['R1']:
# Alternative: Use as resistance level for selling
# Category: PIVOT | Strength: BULLISH

# SIGNAL #95: BELOW S1 PIVOT
# Calculation: Close crosses below S1 support
# Code: if current['Close'] < current['S1'] and 
#          prev['Close'] >= prev['S1']:
# Alternative: Use as support level for buying
# Category: PIVOT | Strength: BEARISH

# SIGNAL #96: AT FIB 61.8%
# Calculation: Close within 1% of 61.8% Fibonacci retracement
# Fibonacci Calculation:
#   50-period High and Low
#   Diff = High - Low
#   Fib_618 = High - (0.618 * Diff)
# Code: if abs(current['Close'] - current['Fib_618']) / current['Close'] < 0.01:
# Alternative: Use 2% tolerance for more frequent signals
# Alternative: Use 0.5% for tighter levels
# Category: FIBONACCI | Strength: SIGNIFICANT

# SIGNAL #97: AT FIB 50%
# Calculation: Close within 1% of 50% retracement
# Code: if abs(current['Close'] - current['Fib_500']) / current['Close'] < 0.01:
# Alternative: Use 2% tolerance
# Category: FIBONACCI | Strength: SIGNIFICANT

# ============================================================================
# 10B. MULTI-TIMEFRAME FIBONACCI RETRACEMENT SIGNALS (30 NEW SIGNALS)
# ============================================================================
# These signals detect price at key Fibonacci levels across different periods
# Fibonacci Retracement Levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%
# Each level is calculated for: 2W, 1M, 3M, 6M, 1Y, 2Y lookback periods

# 1-DAY FIBONACCI SIGNALS (4 signals)
# ====================================

# SIGNAL #98: PRICE AT 1-DAY FIB 38.2%
# Calculation:
#   Look back 1 day (yesterday's high/low)
#   high_1d = df['High'].iloc[-1]
#   low_1d = df['Low'].iloc[-1]
#   fib_382_1d = high_1d - (0.382 * (high_1d - low_1d))
# Code:
#   if abs(current['Close'] - fib_382_1d) / current['Close'] < 0.01:
# Interpretation: Intraday retracement - very short-term
# Category: FIBONACCI_1D | Strength: MODERATE

# SIGNAL #99: PRICE AT 1-DAY FIB 50%
# Calculation:
#   fib_50_1d = high_1d - (0.5 * (high_1d - low_1d))
# Code:
#   if abs(current['Close'] - fib_50_1d) / current['Close'] < 0.01:
# Interpretation: Midpoint of yesterday's range
# Category: FIBONACCI_1D | Strength: SIGNIFICANT

# SIGNAL #100: PRICE AT 1-DAY FIB 61.8%
# Calculation:
#   fib_618_1d = high_1d - (0.618 * (high_1d - low_1d))
# Code:
#   if abs(current['Close'] - fib_618_1d) / current['Close'] < 0.01:
# Interpretation: Golden ratio intraday level
# Category: FIBONACCI_1D | Strength: SIGNIFICANT

# SIGNAL #101: INTRADAY BOUNCE FROM 1-DAY FIB 50%
# Calculation: Price crosses above fib_50_1d
# Code:
#   if prev['Close'] <= fib_50_1d and current['Close'] > fib_50_1d:
# Interpretation: Quick reversal from midpoint
# Category: FIBONACCI_1D | Strength: BULLISH

# 5-DAY FIBONACCI SIGNALS (5 signals)
# ===================================

# SIGNAL #102: PRICE AT 5-DAY FIB 23.6%
# Calculation:
#   Look back 5 days
#   high_5d = df['High'].iloc[-5:].max()
#   low_5d = df['Low'].iloc[-5:].min()
#   fib_236_5d = high_5d - (0.236 * (high_5d - low_5d))
# Code:
#   if abs(current['Close'] - fib_236_5d) / current['Close'] < 0.01:
# Interpretation: Shallow weekly retracement
# Category: FIBONACCI_5D | Strength: MODERATE

# SIGNAL #103: PRICE AT 5-DAY FIB 38.2%
# Calculation:
#   fib_382_5d = high_5d - (0.382 * (high_5d - low_5d))
# Code:
#   if abs(current['Close'] - fib_382_5d) / current['Close'] < 0.01:
# Interpretation: Common short-term retracement
# Category: FIBONACCI_5D | Strength: MODERATE

# SIGNAL #104: PRICE AT 5-DAY FIB 50%
# Calculation:
#   fib_50_5d = high_5d - (0.5 * (high_5d - low_5d))
# Code:
#   if abs(current['Close'] - fib_50_5d) / current['Close'] < 0.01:
# Interpretation: Midpoint of 5-day range
# Category: FIBONACCI_5D | Strength: SIGNIFICANT

# SIGNAL #105: PRICE AT 5-DAY FIB 61.8%
# Calculation:
#   fib_618_5d = high_5d - (0.618 * (high_5d - low_5d))
# Code:
#   if abs(current['Close'] - fib_618_5d) / current['Close'] < 0.01:
# Interpretation: Major short-term support/resistance
# Category: FIBONACCI_5D | Strength: SIGNIFICANT

# SIGNAL #106: PRICE BETWEEN 5-DAY FIB 38.2% AND 50%
# Calculation: Close is in early retracement zone
# Code:
#   if fib_382_5d <= current['Close'] <= fib_50_5d:
# Interpretation: Pullback within first half
# Category: FIBONACCI_5D | Strength: MODERATE

# 1-WEEK FIBONACCI SIGNALS (5 signals)
# ===================================

# SIGNAL #107: PRICE AT 1-WEEK FIB 23.6%
# Calculation:
#   Look back 7 days (1 week)
#   high_1w = df['High'].iloc[-7:].max()
#   low_1w = df['Low'].iloc[-7:].min()
#   fib_236_1w = high_1w - (0.236 * (high_1w - low_1w))
# Code:
#   if abs(current['Close'] - fib_236_1w) / current['Close'] < 0.01:
# Interpretation: Shallow weekly retracement
# Category: FIBONACCI_1W | Strength: MODERATE

# SIGNAL #108: PRICE AT 1-WEEK FIB 38.2%
# Calculation:
#   fib_382_1w = high_1w - (0.382 * (high_1w - low_1w))
# Code:
#   if abs(current['Close'] - fib_382_1w) / current['Close'] < 0.01:
# Interpretation: First major weekly retracement
# Category: FIBONACCI_1W | Strength: MODERATE

# SIGNAL #109: PRICE AT 1-WEEK FIB 50%
# Calculation:
#   fib_50_1w = high_1w - (0.5 * (high_1w - low_1w))
# Code:
#   if abs(current['Close'] - fib_50_1w) / current['Close'] < 0.01:
# Interpretation: Midpoint of weekly range
# Category: FIBONACCI_1W | Strength: SIGNIFICANT

# SIGNAL #110: PRICE AT 1-WEEK FIB 61.8%
# Calculation:
#   fib_618_1w = high_1w - (0.618 * (high_1w - low_1w))
# Code:
#   if abs(current['Close'] - fib_618_1w) / current['Close'] < 0.01:
# Interpretation: Golden ratio weekly level
# Category: FIBONACCI_1W | Strength: SIGNIFICANT

# SIGNAL #111: PRICE BOUNCING FROM 1-WEEK FIB 61.8%
# Calculation: Close crosses above fib_618_1w
# Code:
#   if prev['Close'] <= fib_618_1w and current['Close'] > fib_618_1w:
# Interpretation: Weekly trend resumption
# Category: FIBONACCI_1W | Strength: STRONG BULLISH

# 2-WEEK FIBONACCI SIGNALS (5 signals)
# =====================================

# SIGNAL #112: PRICE AT 2-WEEK FIB 38.2%
# Calculation: 
#   Look back 14 days, find highest high and lowest low
#   Fib_38.2 = High - (0.382 * (High - Low))
#   Trigger: Close within 1% of this level
# Code needed:
#   high_2w = df['High'].iloc[-14:].max()
#   low_2w = df['Low'].iloc[-14:].min()
#   fib_38_2w = high_2w - (0.382 * (high_2w - low_2w))
#   if abs(current['Close'] - fib_38_2w) / current['Close'] < 0.01:
# Alternative: Use 2% tolerance for more frequent signals
# Alternative: Add volume confirmation when touching level
# Alternative: Require price to touch and bounce (not break through)
# Interpretation: Common retracement level - expect support/resistance
# Category: FIBONACCI_2W | Strength: MODERATE

# SIGNAL #99: PRICE AT 2-WEEK FIB 50%
# Calculation: 
#   Fib_50 = High - (0.5 * (High - Low))
# Code:
#   fib_50_2w = high_2w - (0.5 * (high_2w - low_2w))
# Alternative: Use as key support/resistance level
# Interpretation: Major psychological and technical level
# Category: FIBONACCI_2W | Strength: SIGNIFICANT

# SIGNAL #100: PRICE AT 2-WEEK FIB 61.8%
# Calculation:
#   Fib_61.8 = High - (0.618 * (High - Low))
# Code:
#   fib_618_2w = high_2w - (0.618 * (high_2w - low_2w))
# Alternative: Golden ratio - most important level
# Interpretation: Strongest retracement level, major reversal zone
# Category: FIBONACCI_2W | Strength: SIGNIFICANT

# SIGNAL #101: PRICE BELOW 2-WEEK FIB 38.2%
# Calculation: Close < Fib_38.2 (breakthrough into deeper retracement)
# Code:
#   if current['Close'] < fib_38_2w * 0.99:  # 1% below
# Alternative: Trigger when crossing below (prev >= level, current < level)
# Interpretation: Price reverting deeper than expected
# Category: FIBONACCI_2W | Strength: BEARISH

# SIGNAL #102: PRICE ABOVE 2-WEEK FIB 61.8%
# Calculation: Close > Fib_61.8 (breakthrough to new territory)
# Code:
#   if current['Close'] > fib_618_2w * 1.01:
# Alternative: Require high volume with breakout
# Interpretation: Failed retracement - continuing original move
# Category: FIBONACCI_2W | Strength: BULLISH

# 1-MONTH FIBONACCI SIGNALS (5 signals)
# =====================================

# SIGNAL #103: PRICE AT 1-MONTH FIB 23.6%
# Calculation:
#   Look back 21 days (1 month)
#   high_1m = df['High'].iloc[-21:].max()
#   low_1m = df['Low'].iloc[-21:].min()
#   fib_236_1m = high_1m - (0.236 * (high_1m - low_1m))
# Code:
#   if abs(current['Close'] - fib_236_1m) / current['Close'] < 0.01:
# Alternative: Use 0.5% tolerance for precise level
# Interpretation: Shallow retracement - early support
# Category: FIBONACCI_1M | Strength: MODERATE

# SIGNAL #104: PRICE AT 1-MONTH FIB 38.2%
# Calculation:
#   fib_382_1m = high_1m - (0.382 * (high_1m - low_1m))
# Code:
#   if abs(current['Close'] - fib_382_1m) / current['Close'] < 0.01:
# Interpretation: Common retracement zone
# Category: FIBONACCI_1M | Strength: MODERATE

# SIGNAL #105: PRICE AT 1-MONTH FIB 50%
# Calculation:
#   fib_50_1m = high_1m - (0.5 * (high_1m - low_1m))
# Code:
#   if abs(current['Close'] - fib_50_1m) / current['Close'] < 0.01:
# Interpretation: Exact midpoint - psychological level
# Category: FIBONACCI_1M | Strength: SIGNIFICANT

# SIGNAL #106: PRICE AT 1-MONTH FIB 61.8%
# Calculation:
#   fib_618_1m = high_1m - (0.618 * (high_1m - low_1m))
# Code:
#   if abs(current['Close'] - fib_618_1m) / current['Close'] < 0.01:
# Interpretation: Golden ratio - strongest retracement
# Category: FIBONACCI_1M | Strength: SIGNIFICANT

# SIGNAL #107: PRICE BETWEEN 1-MONTH FIB 38.2% AND 50%
# Calculation: Close is between fib_382_1m and fib_50_1m
# Code:
#   if fib_382_1m <= current['Close'] <= fib_50_1m:
# Interpretation: Shallow retracement zone - potential bounce
# Category: FIBONACCI_1M | Strength: MODERATE

# 3-MONTH FIBONACCI SIGNALS (5 signals)
# =====================================

# SIGNAL #108: PRICE AT 3-MONTH FIB 23.6%
# Calculation:
#   Look back 63 days (3 months)
#   high_3m = df['High'].iloc[-63:].max()
#   low_3m = df['Low'].iloc[-63:].min()
#   fib_236_3m = high_3m - (0.236 * (high_3m - low_3m))
# Code:
#   if abs(current['Close'] - fib_236_3m) / current['Close'] < 0.01:
# Interpretation: Very shallow retracement on longer timeframe
# Category: FIBONACCI_3M | Strength: MODERATE

# SIGNAL #109: PRICE AT 3-MONTH FIB 38.2%
# Calculation:
#   fib_382_3m = high_3m - (0.382 * (high_3m - low_3m))
# Code:
#   if abs(current['Close'] - fib_382_3m) / current['Close'] < 0.01:
# Interpretation: Common retracement on quarterly timeframe
# Category: FIBONACCI_3M | Strength: MODERATE

# SIGNAL #110: PRICE AT 3-MONTH FIB 50%
# Calculation:
#   fib_50_3m = high_3m - (0.5 * (high_3m - low_3m))
# Code:
#   if abs(current['Close'] - fib_50_3m) / current['Close'] < 0.01:
# Interpretation: Midpoint of 3-month range
# Category: FIBONACCI_3M | Strength: SIGNIFICANT

# SIGNAL #111: PRICE AT 3-MONTH FIB 61.8%
# Calculation:
#   fib_618_3m = high_3m - (0.618 * (high_3m - low_3m))
# Code:
#   if abs(current['Close'] - fib_618_3m) / current['Close'] < 0.01:
# Interpretation: Major retracement zone - strong support/resistance
# Category: FIBONACCI_3M | Strength: SIGNIFICANT

# SIGNAL #112: PRICE BOUNCING FROM 3-MONTH FIB 61.8%
# Calculation: Close crosses above fib_618_3m (yesterday below, today above)
# Code:
#   if prev['Close'] <= fib_618_3m and current['Close'] > fib_618_3m:
# Interpretation: Major uptrend resumption after retracement
# Category: FIBONACCI_3M | Strength: STRONG BULLISH

# 6-MONTH FIBONACCI SIGNALS (5 signals)
# =====================================

# SIGNAL #113: PRICE AT 6-MONTH FIB 23.6%
# Calculation:
#   Look back 126 days (6 months)
#   high_6m = df['High'].iloc[-126:].max()
#   low_6m = df['Low'].iloc[-126:].min()
#   fib_236_6m = high_6m - (0.236 * (high_6m - low_6m))
# Code:
#   if abs(current['Close'] - fib_236_6m) / current['Close'] < 0.01:
# Interpretation: Shallow retracement on semi-annual range
# Category: FIBONACCI_6M | Strength: MODERATE

# SIGNAL #114: PRICE AT 6-MONTH FIB 38.2%
# Calculation:
#   fib_382_6m = high_6m - (0.382 * (high_6m - low_6m))
# Code:
#   if abs(current['Close'] - fib_382_6m) / current['Close'] < 0.01:
# Interpretation: Common mid-range retracement
# Category: FIBONACCI_6M | Strength: MODERATE

# SIGNAL #115: PRICE AT 6-MONTH FIB 50%
# Calculation:
#   fib_50_6m = high_6m - (0.5 * (high_6m - low_6m))
# Code:
#   if abs(current['Close'] - fib_50_6m) / current['Close'] < 0.01:
# Interpretation: Exact midpoint of 6-month movement
# Category: FIBONACCI_6M | Strength: SIGNIFICANT

# SIGNAL #116: PRICE AT 6-MONTH FIB 61.8%
# Calculation:
#   fib_618_6m = high_6m - (0.618 * (high_6m - low_6m))
# Code:
#   if abs(current['Close'] - fib_618_6m) / current['Close'] < 0.01:
# Interpretation: Golden ratio on semi-annual range
# Category: FIBONACCI_6M | Strength: SIGNIFICANT

# SIGNAL #117: PRICE BETWEEN 6-MONTH FIB 50% AND 61.8%
# Calculation: Close is in critical retracement zone
# Code:
#   if fib_50_6m >= current['Close'] >= fib_618_6m:
# Interpretation: Deep retracement zone - major decision point
# Category: FIBONACCI_6M | Strength: SIGNIFICANT

# 1-YEAR FIBONACCI SIGNALS (5 signals)
# ====================================

# SIGNAL #118: PRICE AT 1-YEAR FIB 23.6%
# Calculation:
#   Look back 252 days (1 year trading)
#   high_1y = df['High'].iloc[-252:].max()
#   low_1y = df['Low'].iloc[-252:].min()
#   fib_236_1y = high_1y - (0.236 * (high_1y - low_1y))
# Code:
#   if abs(current['Close'] - fib_236_1y) / current['Close'] < 0.01:
# Interpretation: Minor pullback on annual range
# Category: FIBONACCI_1Y | Strength: MODERATE

# SIGNAL #119: PRICE AT 1-YEAR FIB 38.2%
# Calculation:
#   fib_382_1y = high_1y - (0.382 * (high_1y - low_1y))
# Code:
#   if abs(current['Close'] - fib_382_1y) / current['Close'] < 0.01:
# Interpretation: First major retracement zone on yearly chart
# Category: FIBONACCI_1Y | Strength: MODERATE

# SIGNAL #120: PRICE AT 1-YEAR FIB 50%
# Calculation:
#   fib_50_1y = high_1y - (0.5 * (high_1y - low_1y))
# Code:
#   if abs(current['Close'] - fib_50_1y) / current['Close'] < 0.01:
# Interpretation: Midpoint of entire year's range
# Category: FIBONACCI_1Y | Strength: SIGNIFICANT

# SIGNAL #121: PRICE AT 1-YEAR FIB 61.8%
# Calculation:
#   fib_618_1y = high_1y - (0.618 * (high_1y - low_1y))
# Code:
#   if abs(current['Close'] - fib_618_1y) / current['Close'] < 0.01:
# Interpretation: Critical long-term support/resistance
# Category: FIBONACCI_1Y | Strength: SIGNIFICANT

# SIGNAL #122: PRICE AT 1-YEAR FIB 78.6%
# Calculation:
#   fib_786_1y = high_1y - (0.786 * (high_1y - low_1y))
# Code:
#   if abs(current['Close'] - fib_786_1y) / current['Close'] < 0.01:
# Interpretation: Very deep retracement - near extremes
# Category: FIBONACCI_1Y | Strength: SIGNIFICANT

# 2-YEAR FIBONACCI SIGNALS (5 signals)
# ====================================

# SIGNAL #123: PRICE AT 2-YEAR FIB 23.6%
# Calculation:
#   Look back 504 days (2 years trading)
#   high_2y = df['High'].iloc[-504:].max()
#   low_2y = df['Low'].iloc[-504:].min()
#   fib_236_2y = high_2y - (0.236 * (high_2y - low_2y))
# Code:
#   if abs(current['Close'] - fib_236_2y) / current['Close'] < 0.01:
# Interpretation: Shallow retracement on multi-year range
# Category: FIBONACCI_2Y | Strength: MODERATE

# SIGNAL #124: PRICE AT 2-YEAR FIB 38.2%
# Calculation:
#   fib_382_2y = high_2y - (0.382 * (high_2y - low_2y))
# Code:
#   if abs(current['Close'] - fib_382_2y) / current['Close'] < 0.01:
# Interpretation: Major retracement zone on 2-year chart
# Category: FIBONACCI_2Y | Strength: SIGNIFICANT

# SIGNAL #125: PRICE AT 2-YEAR FIB 50%
# Calculation:
#   fib_50_2y = high_2y - (0.5 * (high_2y - low_2y))
# Code:
#   if abs(current['Close'] - fib_50_2y) / current['Close'] < 0.01:
# Interpretation: Exact midpoint of 2-year movement
# Category: FIBONACCI_2Y | Strength: SIGNIFICANT

# SIGNAL #126: PRICE AT 2-YEAR FIB 61.8%
# Calculation:
#   fib_618_2y = high_2y - (0.618 * (high_2y - low_2y))
# Code:
#   if abs(current['Close'] - fib_618_2y) / current['Close'] < 0.01:
# Interpretation: Golden ratio on extended timeframe
# Category: FIBONACCI_2Y | Strength: SIGNIFICANT

# SIGNAL #127: MULTI-TIMEFRAME FIB CONVERGENCE
# Calculation: Price touching multiple Fibonacci levels simultaneously
# Code:
#   at_fib_2w = any([abs(current['Close'] - fib) < current['Close']*0.01 
#                    for fib in [fib_236_2w, fib_382_2w, fib_50_2w, fib_618_2w]])
#   at_fib_1m = any([similar for 1m fibs])
#   at_fib_3m = any([similar for 3m fibs])
#   if (at_fib_2w + at_fib_1m + at_fib_3m) >= 2:  # At least 2 timeframes
# Interpretation: Extremely strong support/resistance cluster
# Category: FIBONACCI_CONVERGENCE | Strength: EXTREME

# ============================================================================
# FIBONACCI IMPLEMENTATION CODE TEMPLATE
# ============================================================================

"""
# Add to calculate_indicators() method:

# Multi-timeframe Fibonacci calculations
periods = {
    '2W': 14,
    '1M': 21,
    '3M': 63,
    '6M': 126,
    '1Y': 252,
    '2Y': 504
}

for period_name, period_days in periods.items():
    if len(df) >= period_days:
        high_period = df['High'].iloc[-period_days:].max()
        low_period = df['Low'].iloc[-period_days:].min()
        diff = high_period - low_period
        
        df[f'Fib_236_{period_name}'] = high_period - (0.236 * diff)
        df[f'Fib_382_{period_name}'] = high_period - (0.382 * diff)
        df[f'Fib_500_{period_name}'] = high_period - (0.5 * diff)
        df[f'Fib_618_{period_name}'] = high_period - (0.618 * diff)
        df[f'Fib_786_{period_name}'] = high_period - (0.786 * diff)

# Add to detect_signals() method:

fib_signals = []

# Check each timeframe
for period_name in ['2W', '1M', '3M', '6M', '1Y', '2Y']:
    fib_levels = {
        '23.6%': f'Fib_236_{period_name}',
        '38.2%': f'Fib_382_{period_name}',
        '50%': f'Fib_500_{period_name}',
        '61.8%': f'Fib_618_{period_name}',
        '78.6%': f'Fib_786_{period_name}'
    }
    
    for level_name, col_name in fib_levels.items():
        if col_name in current.index and not pd.isna(current[col_name]):
            fib_level = current[col_name]
            tolerance = abs(current['Close'] * 0.01)  # 1% tolerance
            
            if abs(current['Close'] - fib_level) < tolerance:
                signals.append({
                    'signal': f'{period_name} FIB {level_name}',
                    'desc': f'Price at Fibonacci {level_name} ({period_name})',
                    'strength': 'SIGNIFICANT' if level_name in ['50%', '61.8%'] else 'MODERATE',
                    'category': f'FIBONACCI_{period_name}'
                })
            
            # Check for breakouts
            if len(df) > 1 and col_name in prev.index:
                prev_fib = prev[col_name]
                if prev['Close'] <= fib_level and current['Close'] > fib_level:
                    signals.append({
                        'signal': f'{period_name} FIB {level_name} BREAKOUT',
                        'desc': f'Price broke above {level_name} Fib ({period_name})',
                        'strength': 'STRONG BULLISH',
                        'category': f'FIBONACCI_{period_name}'
                    })
                elif prev['Close'] >= fib_level and current['Close'] < fib_level:
                    signals.append({
                        'signal': f'{period_name} FIB {level_name} BREAKDOWN',
                        'desc': f'Price broke below {level_name} Fib ({period_name})',
                        'strength': 'STRONG BEARISH',
                        'category': f'FIBONACCI_{period_name}'
                    })
"""

# ============================================================================
# 11. MULTI-INDICATOR & VOLATILITY SIGNALS
# ============================================================================

# SIGNAL #98: HIGH VOLATILITY
# Calculation: Annualized Volatility > 40%
# Volatility = StdDev(Daily %) * sqrt(252) * 100
# Code: if current['Volatility'] > 40:
# Alternative: Use 35% for more frequent signals
# Alternative: Use percentile ranking (e.g., > 90th percentile)
# Category: VOLATILITY | Strength: SIGNIFICANT

# SIGNAL #99: LOW VOLATILITY
# Calculation: Annualized Volatility < 15%
# Code: if current['Volatility'] < 15:
# Alternative: Use 20% for less stringent
# Alternative: Use percentile (< 25th)
# Category: VOLATILITY_SQUEEZE | Strength: NEUTRAL

# SIGNAL #100: VOLATILITY EXPANSION
# Calculation: Volatility increased > 50% over 10 days
# Code: vol_change = (current['Volatility'] - df['Volatility'].iloc[-10]) / 
#                    df['Volatility'].iloc[-10] * 100
#       if vol_change > 50:
# Alternative: Use 30% threshold
# Alternative: Use absolute vol increase (e.g., > 10 percentage points)
# Category: VOLATILITY | Strength: SIGNIFICANT

# SIGNAL #101: TIGHT CONSOLIDATION
# Calculation: Price range < 5% over 10 days
# Code: price_range = (df['High'].iloc[-10:].max() - df['Low'].iloc[-10:].min()) / 
#                     df['Close'].iloc[-10:].mean() * 100
#       if price_range < 5:
# Alternative: Use 3% for tighter consolidation
# Alternative: Use Bollinger Band Width as alternative
# Category: CONSOLIDATION | Strength: NEUTRAL

# SIGNAL #102: MOMENTUM ACCELERATION (Bullish)
# Calculation: Momentum increasing AND Momentum > 0
# Momentum = Close - Close[10]
# Code: if current['Momentum'] > prev['Momentum'] > prev2['Momentum'] and 
#          current['Momentum'] > 0:
# Alternative: Use Rate of Change instead of Momentum
# Alternative: Require 4+ consecutive days of acceleration
# Category: MOMENTUM_ACCELERATION | Strength: STRONG BULLISH

# SIGNAL #103: MOMENTUM DECELERATION (Bearish)
# Calculation: Momentum decreasing AND Momentum < 0
# Code: elif current['Momentum'] < prev['Momentum'] < prev2['Momentum'] and 
#           current['Momentum'] < 0:
# Alternative: Use ROC
# Category: MOMENTUM_ACCELERATION | Strength: STRONG BEARISH

# SIGNAL #104: MOMENTUM DIVERGENCE
# Calculation: Price up 10+ days, ROC < -20%
# Code: price_roc = (current['Close'] - df['Close'].iloc[-10]) / 
#                   df['Close'].iloc[-10] * 100
#       volume_roc = (current['Volume'] - df['Volume'].iloc[-10]) / 
#                    df['Volume'].iloc[-10] * 100
#       if price_roc > 5 and volume_roc < -20:
# Alternative: Use price up but MACD down
# Alternative: Use price up but RSI down
# Category: MOMENTUM_DIVERGENCE | Strength: BEARISH

# SIGNAL #105: FULL TIMEFRAME ALIGNMENT (Bullish)
# Calculation: Close > SMA_10 AND Close > SMA_50 AND Close > SMA_200
# Code: short_term_bull = current['Close'] > current['SMA_10']
#       mid_term_bull = current['Close'] > current['SMA_50']
#       long_term_bull = current['Close'] > current['SMA_200']
#       if short_term_bull and mid_term_bull and long_term_bull:
# Alternative: Use all EMAs for faster response
# Alternative: Require MAs also in alignment (10 > 50 > 200)
# Category: TIMEFRAME_ALIGNMENT | Strength: EXTREME BULLISH

# SIGNAL #106: FULL TIMEFRAME ALIGNMENT (Bearish)
# Calculation: Close < SMA_10 AND Close < SMA_50 AND Close < SMA_200
# Code: if not short_term_bull and not mid_term_bull and not long_term_bull:
# Alternative: Use EMAs
# Category: TIMEFRAME_ALIGNMENT | Strength: EXTREME BEARISH

# SIGNAL #107: MULTI-INDICATOR BULLISH
# Calculation: 4+ of 5 indicators bullish
# Indicators: RSI > 50, MACD > Signal, Close > SMA_20, Stoch_K > Stoch_D, +DI > -DI
# Code: bullish_count = sum([current['RSI'] > 50, 
#                            current['MACD'] > current['MACD_Signal'],
#                            current['Close'] > current['SMA_20'],
#                            current['Stoch_K'] > current['Stoch_D'],
#                            current['Plus_DI'] > current['Minus_DI']])
#       if bullish_count >= 4:
# Alternative: Use 5/5 for stronger signal
# Alternative: Add additional indicators (CCI > 0, Williams %R > -50, etc.)
# Category: MULTI_INDICATOR | Strength: STRONG BULLISH

# SIGNAL #108: MULTI-INDICATOR BEARISH
# Calculation: 4+ of 5 indicators bearish (inverted from above)
# Code: elif bullish_count <= 1:
# Alternative: Use 0/5 for stronger signal
# Category: MULTI_INDICATOR | Strength: STRONG BEARISH

# ============================================================================
# ADDITIONAL REFERENCE NOTES
# ============================================================================

"""
CALCULATION METHOD VARIATIONS WORTH CONSIDERING:

1. RSI ALTERNATIVE (Smooth RSI):
   - Use Wilder's smoothing: alpha = 1/14
   - Current method uses simple rolling mean which differs slightly
   
2. MOVING AVERAGES ALTERNATIVE:
   - Hull Moving Average (HMA) = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
   - Exponential Hull Moving Average for even faster response
   
3. BOLLINGER BANDS ALTERNATIVES:
   - Keltner Channels (ATR-based instead of StdDev)
   - Donchian Channels (High/Low of period instead of price-based)
   
4. VOLUME-WEIGHTED PRICE ALTERNATIVES:
   - VWMA (Volume Weighted Moving Average) instead of simple MA
   - Uses actual weighted calculations
   
5. ATR CALCULATION:
   - Current uses simple 14-period MA
   - Alternative: Wilder's smoothing (exponential) for more accuracy
   
6. ADX CALCULATION:
   - Current implementation appears standard
   - Consider Wilder's smoothing for more traditional approach
   
7. STOCHASTIC ALTERNATIVES:
   - Fast Stochastic (K only, no D)
   - Slow Stochastic (D line, more smoothed)
   - Current uses standard (14, 3) settings
   
8. MACD ALTERNATIVES:
   - Fast MACD (5, 35, 5) for quicker signals
   - Slow MACD (12, 26, 9) - current standard
   - MACD histogram divergence (more advanced)

9. ICHIMOKU IMPROVEMENTS:
   - Adjust periods for different timeframes
   - Use Chikou Span for additional confirmation
   
10. DIVERGENCE DETECTION:
    - Use linear regression instead of fixed lookback period
    - Identify peaks/troughs automatically rather than fixed 20-day window

SIGNAL STRENGTH GUIDELINES:
- EXTREME: 85-100 points (major reversals, multi-indicator confluences)
- STRONG: 70-84 points (clear, established patterns)
- MODERATE: 55-69 points (interesting, but needs confirmation)
- WEAK: 40-54 points (potential, but require additional confluence)
- NOISE: 0-39 points (likely false signals without confluence)

CATEGORY RELIABILITY RANKINGS:
1. MA_CROSS (Golden/Death Cross) - Highly reliable, especially with confluence
2. STRUCTURE_BREAK - Clear breakouts of established levels
3. TIMEFRAME_ALIGNMENT - Multiple-timeframe agreement
4. MULTI_INDICATOR - 4+ indicators aligned
5. ADX_BASED - Trend strength confirmation
6. BB_WALK - Price action with technical confirmation
7. CANDLESTICK - Specific formations with historical testing
8. VOLUME_BASED - Critical for move confirmation
9. DIVERGENCE - Advanced, requires proper identification
10. SINGLE_INDICATOR - Need confluence with other signals
"""