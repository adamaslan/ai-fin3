"""
Advanced Technical Analysis Scanner with 100 Alerts
Comprehensive monitoring of price action, patterns, volume, and momentum
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TechnicalAnalyzer:
    def __init__(self, symbol, period='1y'):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.indicators = {}
        self.signals = {}
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        print(f"📊 Fetching data for {self.symbol}...")
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(period=self.period)
        
        if self.data.empty:
            raise ValueError(f"No data found for {self.symbol}")
        
        print(f"✅ Fetched {len(self.data)} days of data")
        return self.data
    
    def calculate_indicators(self):
        """Calculate comprehensive technical indicators"""
        df = self.data.copy()
        
        print("\n🔧 Calculating Technical Indicators...")
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Stochastic
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # ADX
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr14 = true_range.rolling(14).sum()
        plus_di = 100 * (plus_dm.rolling(14).sum() / tr14)
        minus_di = 100 * (minus_dm.rolling(14).sum() / tr14)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(14).mean()
        df['Plus_DI'] = plus_di
        df['Minus_DI'] = minus_di
        
        # CCI
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        
        # Williams %R
        df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
        
        # OBV
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Volume
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_MA_50'] = df['Volume'].rolling(window=50).mean()
        
        # VWAP
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        # ROC
        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
        
        # MFI
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        mfi_ratio = positive_flow / negative_flow
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        
        # Ichimoku
        high_9 = df['High'].rolling(window=9).max()
        low_9 = df['Low'].rolling(window=9).min()
        df['Tenkan'] = (high_9 + low_9) / 2
        high_26 = df['High'].rolling(window=26).max()
        low_26 = df['Low'].rolling(window=26).min()
        df['Kijun'] = (high_26 + low_26) / 2
        df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
        high_52 = df['High'].rolling(window=52).max()
        low_52 = df['Low'].rolling(window=52).min()
        df['Senkou_B'] = ((high_52 + low_52) / 2).shift(26)
        
        # Momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        
        # Volatility
        df['Volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change() * 100
        df['Price_Change_5d'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
        
        # Highs and Lows
        df['High_52w'] = df['High'].rolling(window=252).max()
        df['Low_52w'] = df['Low'].rolling(window=252).min()
        df['High_20d'] = df['High'].rolling(window=20).max()
        df['Low_20d'] = df['Low'].rolling(window=20).min()
        
        # Distance from MAs
        for period in [10, 20, 50, 200]:
            df[f'Dist_SMA_{period}'] = ((df['Close'] - df[f'SMA_{period}']) / df[f'SMA_{period}']) * 100
        
        self.data = df
        print("✅ All indicators calculated")
        return df
    
    def detect_signals(self):
        """Detect 100 comprehensive technical signals"""
        df = self.data.copy()
        current = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3] if len(df) > 2 else prev
        
        signals = []
        
        print("\n🎯 Scanning for 100 Technical Alerts...")
        print("=" * 80)
        
        # === MOVING AVERAGE CROSSOVERS (10 alerts) ===
        
        # 1. Golden Cross
        if len(df) > 200 and prev['SMA_50'] <= prev['SMA_200'] and current['SMA_50'] > current['SMA_200']:
            signals.append({'signal': '🟢 GOLDEN CROSS', 'desc': '50 MA crossed above 200 MA', 'strength': 'STRONG BULLISH'})
        
        # 2. Death Cross
        if len(df) > 200 and prev['SMA_50'] >= prev['SMA_200'] and current['SMA_50'] < current['SMA_200']:
            signals.append({'signal': '🔴 DEATH CROSS', 'desc': '50 MA crossed below 200 MA', 'strength': 'STRONG BEARISH'})
        
        # 3. Price crosses above 10 MA
        if prev['Close'] <= prev['SMA_10'] and current['Close'] > current['SMA_10']:
            signals.append({'signal': '🟢 PRICE ABOVE 10 MA', 'desc': 'Price crossed above 10-day MA', 'strength': 'BULLISH'})
        
        # 4. Price crosses below 10 MA
        if prev['Close'] >= prev['SMA_10'] and current['Close'] < current['SMA_10']:
            signals.append({'signal': '🔴 PRICE BELOW 10 MA', 'desc': 'Price crossed below 10-day MA', 'strength': 'BEARISH'})
        
        # 5. Price near 10 MA
        if abs(current['Close'] - current['SMA_10']) / current['SMA_10'] < 0.01:
            signals.append({'signal': '⚠️ PRICE AT 10 MA', 'desc': f"Price within 1% of 10 MA (${current['SMA_10']:.2f})", 'strength': 'WATCH'})
        
        # 6. Price crosses above 20 MA
        if prev['Close'] <= prev['SMA_20'] and current['Close'] > current['SMA_20']:
            signals.append({'signal': '🟢 PRICE ABOVE 20 MA', 'desc': 'Price crossed above 20-day MA', 'strength': 'BULLISH'})
        
        # 7. Price crosses below 20 MA
        if prev['Close'] >= prev['SMA_20'] and current['Close'] < current['SMA_20']:
            signals.append({'signal': '🔴 PRICE BELOW 20 MA', 'desc': 'Price crossed below 20-day MA', 'strength': 'BEARISH'})
        
        # 8. 10 EMA crosses 20 EMA (Bullish)
        if prev['EMA_10'] <= prev['EMA_20'] and current['EMA_10'] > current['EMA_20']:
            signals.append({'signal': '🟢 10/20 EMA BULL CROSS', 'desc': '10 EMA crossed above 20 EMA', 'strength': 'BULLISH'})
        
        # 9. 10 EMA crosses 20 EMA (Bearish)
        if prev['EMA_10'] >= prev['EMA_20'] and current['EMA_10'] < current['EMA_20']:
            signals.append({'signal': '🔴 10/20 EMA BEAR CROSS', 'desc': '10 EMA crossed below 20 EMA', 'strength': 'BEARISH'})
        
        # 10. 20 EMA crosses 50 EMA
        if prev['EMA_20'] <= prev['EMA_50'] and current['EMA_20'] > current['EMA_50']:
            signals.append({'signal': '🟢 20/50 EMA BULL CROSS', 'desc': '20 EMA crossed above 50 EMA', 'strength': 'BULLISH'})
        
        # === RSI ALERTS (10 alerts) ===
        
        # 11. RSI Oversold
        if current['RSI'] < 30:
            signals.append({'signal': '🟢 RSI OVERSOLD', 'desc': f"RSI at {current['RSI']:.1f}", 'strength': 'BULLISH'})
        
        # 12. RSI Overbought
        if current['RSI'] > 70:
            signals.append({'signal': '🔴 RSI OVERBOUGHT', 'desc': f"RSI at {current['RSI']:.1f}", 'strength': 'BEARISH'})
        
        # 13. RSI Extremely Oversold
        if current['RSI'] < 20:
            signals.append({'signal': '🟢 RSI EXTREME OVERSOLD', 'desc': f"RSI at {current['RSI']:.1f}", 'strength': 'STRONG BULLISH'})
        
        # 14. RSI Extremely Overbought
        if current['RSI'] > 80:
            signals.append({'signal': '🔴 RSI EXTREME OVERBOUGHT', 'desc': f"RSI at {current['RSI']:.1f}", 'strength': 'STRONG BEARISH'})
        
        # 15. RSI crosses above 50
        if prev['RSI'] <= 50 and current['RSI'] > 50:
            signals.append({'signal': '🟢 RSI ABOVE 50', 'desc': 'RSI crossed bullish threshold', 'strength': 'BULLISH'})
        
        # 16. RSI crosses below 50
        if prev['RSI'] >= 50 and current['RSI'] < 50:
            signals.append({'signal': '🔴 RSI BELOW 50', 'desc': 'RSI crossed bearish threshold', 'strength': 'BEARISH'})
        
        # 17. RSI Divergence Bullish (price lower, RSI higher)
        if len(df) > 20:
            if current['Close'] < df['Close'].iloc[-20] and current['RSI'] > df['RSI'].iloc[-20]:
                signals.append({'signal': '🟢 RSI BULLISH DIVERGENCE', 'desc': 'Price down but RSI up', 'strength': 'BULLISH'})
        
        # 18. RSI Divergence Bearish (price higher, RSI lower)
        if len(df) > 20:
            if current['Close'] > df['Close'].iloc[-20] and current['RSI'] < df['RSI'].iloc[-20]:
                signals.append({'signal': '🔴 RSI BEARISH DIVERGENCE', 'desc': 'Price up but RSI down', 'strength': 'BEARISH'})
        
        # 19. RSI entering oversold zone
        if prev['RSI'] >= 30 and current['RSI'] < 30:
            signals.append({'signal': '🟢 RSI ENTERING OVERSOLD', 'desc': 'RSI just dropped below 30', 'strength': 'WATCH'})
        
        # 20. RSI exiting overbought zone
        if prev['RSI'] >= 70 and current['RSI'] < 70:
            signals.append({'signal': '🔴 RSI EXITING OVERBOUGHT', 'desc': 'RSI dropped from overbought', 'strength': 'BEARISH'})
        
        # === MACD ALERTS (10 alerts) ===
        
        # 21. MACD Bullish Crossover
        if prev['MACD'] <= prev['MACD_Signal'] and current['MACD'] > current['MACD_Signal']:
            signals.append({'signal': '🟢 MACD BULL CROSS', 'desc': 'MACD crossed above signal', 'strength': 'BULLISH'})
        
        # 22. MACD Bearish Crossover
        if prev['MACD'] >= prev['MACD_Signal'] and current['MACD'] < current['MACD_Signal']:
            signals.append({'signal': '🔴 MACD BEAR CROSS', 'desc': 'MACD crossed below signal', 'strength': 'BEARISH'})
        
        # 23. MACD crosses above zero
        if prev['MACD'] <= 0 and current['MACD'] > 0:
            signals.append({'signal': '🟢 MACD ABOVE ZERO', 'desc': 'MACD crossed into positive territory', 'strength': 'BULLISH'})
        
        # 24. MACD crosses below zero
        if prev['MACD'] >= 0 and current['MACD'] < 0:
            signals.append({'signal': '🔴 MACD BELOW ZERO', 'desc': 'MACD crossed into negative territory', 'strength': 'BEARISH'})
        
        # 25. MACD Histogram increasing
        if current['MACD_Hist'] > prev['MACD_Hist'] and prev['MACD_Hist'] > prev2['MACD_Hist']:
            signals.append({'signal': '🟢 MACD MOMENTUM UP', 'desc': 'Histogram expanding bullish', 'strength': 'BULLISH'})
        
        # 26. MACD Histogram decreasing
        if current['MACD_Hist'] < prev['MACD_Hist'] and prev['MACD_Hist'] < prev2['MACD_Hist']:
            signals.append({'signal': '🔴 MACD MOMENTUM DOWN', 'desc': 'Histogram expanding bearish', 'strength': 'BEARISH'})
        
        # 27. MACD Histogram positive and growing
        if current['MACD_Hist'] > 0 and current['MACD_Hist'] > prev['MACD_Hist'] * 1.2:
            signals.append({'signal': '🟢 MACD STRONG MOMENTUM', 'desc': 'Histogram accelerating up', 'strength': 'STRONG BULLISH'})
        
        # 28. MACD Histogram negative and shrinking
        if current['MACD_Hist'] < 0 and current['MACD_Hist'] < prev['MACD_Hist'] * 1.2:
            signals.append({'signal': '🔴 MACD WEAK MOMENTUM', 'desc': 'Histogram accelerating down', 'strength': 'STRONG BEARISH'})
        
        # 29. MACD Signal line trending up
        if current['MACD_Signal'] > prev['MACD_Signal'] and prev['MACD_Signal'] > prev2['MACD_Signal']:
            signals.append({'signal': '🟢 MACD SIGNAL RISING', 'desc': 'Signal line trending upward', 'strength': 'BULLISH'})
        
        # 30. MACD both lines positive
        if current['MACD'] > 0 and current['MACD_Signal'] > 0:
            signals.append({'signal': '🟢 MACD FULLY BULLISH', 'desc': 'Both lines above zero', 'strength': 'BULLISH'})
        
        # === BOLLINGER BANDS (10 alerts) ===
        
        # 31. BB Squeeze
        bb_width_avg = df['BB_Width'].tail(50).mean()
        if current['BB_Width'] < bb_width_avg * 0.7:
            signals.append({'signal': '⚠️ BB SQUEEZE', 'desc': 'Bands narrowing - breakout pending', 'strength': 'NEUTRAL'})
        
        # 32. Price at Lower BB
        if current['Close'] <= current['BB_Lower'] * 1.01:
            signals.append({'signal': '🟢 AT LOWER BB', 'desc': f"Price at ${current['BB_Lower']:.2f}", 'strength': 'BULLISH'})
        
        # 33. Price at Upper BB
        if current['Close'] >= current['BB_Upper'] * 0.99:
            signals.append({'signal': '🔴 AT UPPER BB', 'desc': f"Price at ${current['BB_Upper']:.2f}", 'strength': 'BEARISH'})
        
        # 34. Price broke above Upper BB
        if prev['Close'] <= prev['BB_Upper'] and current['Close'] > current['BB_Upper']:
            signals.append({'signal': '🔴 BROKE UPPER BB', 'desc': 'Strong momentum or overextension', 'strength': 'BEARISH'})
        
        # 35. Price broke below Lower BB
        if prev['Close'] >= prev['BB_Lower'] and current['Close'] < current['BB_Lower']:
            signals.append({'signal': '🟢 BROKE LOWER BB', 'desc': 'Oversold or strong selling', 'strength': 'BULLISH'})
        
        # 36. BB Width expanding
        if current['BB_Width'] > prev['BB_Width'] * 1.1:
            signals.append({'signal': '⚠️ BB EXPANDING', 'desc': 'Volatility increasing', 'strength': 'VOLATILE'})
        
        # 37. Price at BB Middle
        if abs(current['Close'] - current['BB_Middle']) / current['BB_Middle'] < 0.005:
            signals.append({'signal': '⚠️ AT BB MIDDLE', 'desc': 'Price at midline', 'strength': 'NEUTRAL'})
        
        # 38. BB Position very high
        if current['BB_Position'] > 0.95:
            signals.append({'signal': '🔴 BB TOP RANGE', 'desc': f"In top 5% of BB range", 'strength': 'BEARISH'})
        
        # 39. BB Position very low
        if current['BB_Position'] < 0.05:
            signals.append({'signal': '🟢 BB BOTTOM RANGE', 'desc': f"In bottom 5% of BB range", 'strength': 'BULLISH'})
        
        # 40. Walking the bands (upper)
        if current['Close'] > current['BB_Upper'] * 0.98 and prev['Close'] > prev['BB_Upper'] * 0.98:
            signals.append({'signal': '🔴 WALKING UPPER BAND', 'desc': 'Strong uptrend or overbought', 'strength': 'EXTREME'})
        
        # === VOLUME ALERTS (10 alerts) ===
        
        # 41. High Volume Spike
        if current['Volume'] > current['Volume_MA_20'] * 2:
            signals.append({'signal': '📊 VOLUME SPIKE 2X', 'desc': f"Vol: {current['Volume']:,.0f}", 'strength': 'SIGNIFICANT'})
        
        # 42. Extreme Volume Spike
        if current['Volume'] > current['Volume_MA_20'] * 3:
            signals.append({'signal': '📊 EXTREME VOLUME 3X', 'desc': f"Vol: {current['Volume']:,.0f}", 'strength': 'VERY SIGNIFICANT'})
        
        # 43. Low Volume
        if current['Volume'] < current['Volume_MA_20'] * 0.5:
            signals.append({'signal': '📊 LOW VOLUME', 'desc': 'Below average activity', 'strength': 'WEAK'})
        
        # 44. Volume increasing trend
        if current['Volume'] > prev['Volume'] and prev['Volume'] > prev2['Volume']:
            signals.append({'signal': '📊 VOLUME RISING', 'desc': 'Participation increasing', 'strength': 'WATCH'})
        
        # 45. Volume with price up
        if current['Price_Change'] > 2 and current['Volume'] > current['Volume_MA_20'] * 1.5:
            signals.append({'signal': '🟢 VOLUME BREAKOUT', 'desc': 'High volume + price up', 'strength': 'STRONG BULLISH'})
        
        # 46. Volume with price down
        if current['Price_Change'] < -2 and current['Volume'] > current['Volume_MA_20'] * 1.5:
            signals.append({'signal': '🔴 VOLUME SELLOFF', 'desc': 'High volume + price down', 'strength': 'STRONG BEARISH'})
        
        # 47. OBV Rising
        if current['OBV'] > prev['OBV'] and prev['OBV'] > prev2['OBV']:
            signals.append({'signal': '🟢 OBV RISING', 'desc': 'Buying pressure increasing', 'strength': 'BULLISH'})
        
        # 48. OBV Falling
        if current['OBV'] < prev['OBV'] and prev['OBV'] < prev2['OBV']:
            signals.append({'signal': '🔴 OBV FALLING', 'desc': 'Selling pressure increasing', 'strength': 'BEARISH'})
        
        # 49. OBV Divergence Bullish
        if len(df) > 10 and current['Close'] < df['Close'].iloc[-10] and current['OBV'] > df['OBV'].iloc[-10]:
            signals.append({'signal': '🟢 OBV BULL DIVERGENCE', 'desc': 'Price down, OBV up', 'strength': 'BULLISH'})
        
        # 50. Volume drying up
        if current['Volume'] < current['Volume_MA_20'] * 0.3:
            signals.append({'signal': '⚠️ VOLUME DRYING UP', 'desc': 'Very low participation', 'strength': 'CAUTION'})
        
        # === STOCHASTIC ALERTS (5 alerts) ===
        
        # 51. Stochastic Oversold
        if current['Stoch_K'] < 20:
            signals.append({'signal': '🟢 STOCH OVERSOLD', 'desc': f"K: {current['Stoch_K']:.1f}", 'strength': 'BULLISH'})
        
        # 52. Stochastic Overbought
        if current['Stoch_K'] > 80:
            signals.append({'signal': '🔴 STOCH OVERBOUGHT', 'desc': f"K: {current['Stoch_K']:.1f}", 'strength': 'BEARISH'})
        
        # 53. Stochastic Bull Cross
        if prev['Stoch_K'] <= prev['Stoch_D'] and current['Stoch_K'] > current['Stoch_D']:
            signals.append({'signal': '🟢 STOCH BULL CROSS', 'desc': '%K crossed above %D', 'strength': 'BULLISH'})
        
        # 54. Stochastic Bear Cross
        if prev['Stoch_K'] >= prev['Stoch_D'] and current['Stoch_K'] < current['Stoch_D']:
            signals.append({'signal': '🔴 STOCH BEAR CROSS', 'desc': '%K crossed below %D', 'strength': 'BEARISH'})
        
        # 55. Stochastic Exit Oversold
        if prev['Stoch_K'] < 20 and current['Stoch_K'] >= 20:
            signals.append({'signal': '🟢 STOCH EXIT OVERSOLD', 'desc': 'Leaving oversold zone', 'strength': 'BULLISH'})
        
        # === MOMENTUM & TREND (10 alerts) ===
        
        # 56. Strong ADX Trend
        if current['ADX'] > 25:
            trend = 'UP' if current['Close'] > current['SMA_50'] else 'DOWN'
            signals.append({'signal': f"💪 STRONG {trend}TREND", 'desc': f"ADX: {current['ADX']:.1f}", 'strength': 'TRENDING'})
        
        # 57. Very Strong Trend
        if current['ADX'] > 40:
            signals.append({'signal': '💪 VERY STRONG TREND', 'desc': f"ADX: {current['ADX']:.1f}", 'strength': 'EXTREME TREND'})
        
        # 58. Weak Trend
        if current['ADX'] < 20:
            signals.append({'signal': '⚠️ WEAK TREND', 'desc': f"ADX: {current['ADX']:.1f}", 'strength': 'RANGING'})
        
        # 59. +DI crosses above -DI
        if prev['Plus_DI'] <= prev['Minus_DI'] and current['Plus_DI'] > current['Minus_DI']:
            signals.append({'signal': '🟢 +DI ABOVE -DI', 'desc': 'Bullish directional shift', 'strength': 'BULLISH'})
        
        # 60. -DI crosses above +DI
        if prev['Plus_DI'] >= prev['Minus_DI'] and current['Plus_DI'] < current['Minus_DI']:
            signals.append({'signal': '🔴 -DI ABOVE +DI', 'desc': 'Bearish directional shift', 'strength': 'BEARISH'})
        
        # 61. Momentum Positive
        if current['Momentum'] > 0 and prev['Momentum'] <= 0:
            signals.append({'signal': '🟢 MOMENTUM POSITIVE', 'desc': 'Price momentum turning up', 'strength': 'BULLISH'})
        
        # 62. Momentum Negative
        if current['Momentum'] < 0 and prev['Momentum'] >= 0:
            signals.append({'signal': '🔴 MOMENTUM NEGATIVE', 'desc': 'Price momentum turning down', 'strength': 'BEARISH'})
        
        # 63. Strong positive momentum
        if current['Momentum'] > current['ATR'] * 2:
            signals.append({'signal': '🟢 STRONG MOMENTUM UP', 'desc': 'Accelerating upward', 'strength': 'STRONG BULLISH'})
        
        # 64. Strong negative momentum
        if current['Momentum'] < -current['ATR'] * 2:
            signals.append({'signal': '🔴 STRONG MOMENTUM DOWN', 'desc': 'Accelerating downward', 'strength': 'STRONG BEARISH'})
        
        # 65. ROC Extreme Positive
        if current['ROC_10'] > 10:
            signals.append({'signal': '🟢 ROC EXTREME UP', 'desc': f"10-day ROC: {current['ROC_10']:.1f}%", 'strength': 'STRONG BULLISH'})
        
        # === CCI & WILLIAMS %R (5 alerts) ===
        
        # 66. CCI Extreme Oversold
        if current['CCI'] < -200:
            signals.append({'signal': '🟢 CCI EXTREME OVERSOLD', 'desc': f"CCI: {current['CCI']:.1f}", 'strength': 'STRONG BULLISH'})
        
        # 67. CCI Extreme Overbought
        if current['CCI'] > 200:
            signals.append({'signal': '🔴 CCI EXTREME OVERBOUGHT', 'desc': f"CCI: {current['CCI']:.1f}", 'strength': 'STRONG BEARISH'})
        
        # 68. CCI crosses above zero
        if prev['CCI'] <= 0 and current['CCI'] > 0:
            signals.append({'signal': '🟢 CCI ABOVE ZERO', 'desc': 'CCI turned bullish', 'strength': 'BULLISH'})
        
        # 69. Williams %R Oversold
        if current['Williams_R'] < -80:
            signals.append({'signal': '🟢 WILLIAMS OVERSOLD', 'desc': f"W%R: {current['Williams_R']:.1f}", 'strength': 'BULLISH'})
        
        # 70. Williams %R Overbought
        if current['Williams_R'] > -20:
            signals.append({'signal': '🔴 WILLIAMS OVERBOUGHT', 'desc': f"W%R: {current['Williams_R']:.1f}", 'strength': 'BEARISH'})
        
        # === MFI MONEY FLOW (5 alerts) ===
        
        # 71. MFI Oversold
        if current['MFI'] < 20:
            signals.append({'signal': '🟢 MFI OVERSOLD', 'desc': f"MFI: {current['MFI']:.1f}", 'strength': 'BULLISH'})
        
        # 72. MFI Overbought
        if current['MFI'] > 80:
            signals.append({'signal': '🔴 MFI OVERBOUGHT', 'desc': f"MFI: {current['MFI']:.1f}", 'strength': 'BEARISH'})
        
        # 73. MFI crosses 50
        if prev['MFI'] <= 50 and current['MFI'] > 50:
            signals.append({'signal': '🟢 MFI ABOVE 50', 'desc': 'Money flow turning positive', 'strength': 'BULLISH'})
        
        # 74. MFI Extreme Low
        if current['MFI'] < 10:
            signals.append({'signal': '🟢 MFI EXTREME LOW', 'desc': f"MFI: {current['MFI']:.1f}", 'strength': 'STRONG BULLISH'})
        
        # 75. MFI Divergence
        if len(df) > 10 and current['Close'] > df['Close'].iloc[-10] and current['MFI'] < df['MFI'].iloc[-10]:
            signals.append({'signal': '🔴 MFI DIVERGENCE', 'desc': 'Price up, MFI down', 'strength': 'BEARISH'})
        
        # === PRICE ACTION (10 alerts) ===
        
        # 76. Gap Up
        if current['Low'] > prev['High']:
            gap_pct = ((current['Low'] - prev['High']) / prev['High']) * 100
            signals.append({'signal': '🟢 GAP UP', 'desc': f"Gap: {gap_pct:.1f}%", 'strength': 'BULLISH'})
        
        # 77. Gap Down
        if current['High'] < prev['Low']:
            gap_pct = ((prev['Low'] - current['High']) / prev['Low']) * 100
            signals.append({'signal': '🔴 GAP DOWN', 'desc': f"Gap: {gap_pct:.1f}%", 'strength': 'BEARISH'})
        
        # 78. Large Daily Gain
        if current['Price_Change'] > 5:
            signals.append({'signal': '🟢 LARGE GAIN', 'desc': f"+{current['Price_Change']:.1f}% today", 'strength': 'STRONG BULLISH'})
        
        # 79. Large Daily Loss
        if current['Price_Change'] < -5:
            signals.append({'signal': '🔴 LARGE LOSS', 'desc': f"{current['Price_Change']:.1f}% today", 'strength': 'STRONG BEARISH'})
        
        # 80. Three Consecutive Up Days
        if len(df) > 2:
            if current['Close'] > prev['Close'] and prev['Close'] > prev2['Close']:
                signals.append({'signal': '🟢 3 DAYS UP', 'desc': 'Three consecutive green days', 'strength': 'BULLISH'})
        
        # 81. Three Consecutive Down Days
        if len(df) > 2:
            if current['Close'] < prev['Close'] and prev['Close'] < prev2['Close']:
                signals.append({'signal': '🔴 3 DAYS DOWN', 'desc': 'Three consecutive red days', 'strength': 'BEARISH'})
        
        # 82. Doji Pattern
        body = abs(current['Close'] - current['Open'])
        candle_range = current['High'] - current['Low']
        if candle_range > 0 and body / candle_range < 0.1:
            signals.append({'signal': '⚠️ DOJI PATTERN', 'desc': 'Indecision candle', 'strength': 'REVERSAL'})
        
        # 83. Hammer Pattern
        if candle_range > 0:
            lower_shadow = min(current['Open'], current['Close']) - current['Low']
            if lower_shadow > candle_range * 0.6 and body > 0:
                signals.append({'signal': '🟢 HAMMER PATTERN', 'desc': 'Potential bullish reversal', 'strength': 'BULLISH'})
        
        # 84. Shooting Star
        if candle_range > 0:
            upper_shadow = current['High'] - max(current['Open'], current['Close'])
            if upper_shadow > candle_range * 0.6 and body > 0:
                signals.append({'signal': '🔴 SHOOTING STAR', 'desc': 'Potential bearish reversal', 'strength': 'BEARISH'})
        
        # 85. Inside Bar
        if current['High'] < prev['High'] and current['Low'] > prev['Low']:
            signals.append({'signal': '⚠️ INSIDE BAR', 'desc': 'Consolidation - breakout pending', 'strength': 'NEUTRAL'})
        
        # === 52-WEEK & MULTI-PERIOD HIGHS/LOWS (10 alerts) ===
        
        # 86. New 52-Week High
        if current['Close'] >= current['High_52w'] * 0.999:
            signals.append({'signal': '🟢 52-WEEK HIGH', 'desc': f"At ${current['Close']:.2f}", 'strength': 'STRONG BULLISH'})
        
        # 87. New 52-Week Low
        if current['Close'] <= current['Low_52w'] * 1.001:
            signals.append({'signal': '🔴 52-WEEK LOW', 'desc': f"At ${current['Close']:.2f}", 'strength': 'STRONG BEARISH'})
        
        # 88. Near 52-Week High
        if current['Close'] > current['High_52w'] * 0.95 and current['Close'] < current['High_52w']:
            pct_from_high = ((current['High_52w'] - current['Close']) / current['Close']) * 100
            signals.append({'signal': '🟢 NEAR 52W HIGH', 'desc': f"{pct_from_high:.1f}% from high", 'strength': 'BULLISH'})
        
        # 89. Near 52-Week Low
        if current['Close'] < current['Low_52w'] * 1.05 and current['Close'] > current['Low_52w']:
            pct_from_low = ((current['Close'] - current['Low_52w']) / current['Low_52w']) * 100
            signals.append({'signal': '🔴 NEAR 52W LOW', 'desc': f"{pct_from_low:.1f}% from low", 'strength': 'BEARISH'})
        
        # 90. 20-Day High
        if current['Close'] >= current['High_20d'] * 0.999:
            signals.append({'signal': '🟢 20-DAY HIGH', 'desc': 'Short-term breakout', 'strength': 'BULLISH'})
        
        # 91. 20-Day Low
        if current['Close'] <= current['Low_20d'] * 1.001:
            signals.append({'signal': '🔴 20-DAY LOW', 'desc': 'Short-term breakdown', 'strength': 'BEARISH'})
        
        # 92. Price far from 52W high
        pct_from_52w_high = ((current['High_52w'] - current['Close']) / current['Close']) * 100
        if pct_from_52w_high > 50:
            signals.append({'signal': '⚠️ FAR FROM HIGH', 'desc': f"{pct_from_52w_high:.0f}% below 52W high", 'strength': 'WEAK'})
        
        # 93. Recovering from 52W low
        pct_from_52w_low = ((current['Close'] - current['Low_52w']) / current['Low_52w']) * 100
        if pct_from_52w_low > 50 and pct_from_52w_low < 100:
            signals.append({'signal': '🟢 RECOVERING', 'desc': f"{pct_from_52w_low:.0f}% above 52W low", 'strength': 'BULLISH'})
        
        # 94. Multi-week consolidation break up
        if len(df) > 20:
            range_20d = current['High_20d'] - current['Low_20d']
            avg_price = (current['High_20d'] + current['Low_20d']) / 2
            if range_20d / avg_price < 0.1 and current['Close'] > current['High_20d']:
                signals.append({'signal': '🟢 CONSOLIDATION BREAKOUT', 'desc': 'Breaking out of range', 'strength': 'BULLISH'})
        
        # 95. Multi-week consolidation break down
        if len(df) > 20:
            range_20d = current['High_20d'] - current['Low_20d']
            avg_price = (current['High_20d'] + current['Low_20d']) / 2
            if range_20d / avg_price < 0.1 and current['Close'] < current['Low_20d']:
                signals.append({'signal': '🔴 CONSOLIDATION BREAKDOWN', 'desc': 'Breaking down from range', 'strength': 'BEARISH'})
        
        # === ICHIMOKU & ADVANCED (5 alerts) ===
        
        # 96. Tenkan/Kijun Bull Cross
        if prev['Tenkan'] <= prev['Kijun'] and current['Tenkan'] > current['Kijun']:
            signals.append({'signal': '🟢 ICHIMOKU BULL CROSS', 'desc': 'Tenkan above Kijun', 'strength': 'BULLISH'})
        
        # 97. Tenkan/Kijun Bear Cross
        if prev['Tenkan'] >= prev['Kijun'] and current['Tenkan'] < current['Kijun']:
            signals.append({'signal': '🔴 ICHIMOKU BEAR CROSS', 'desc': 'Tenkan below Kijun', 'strength': 'BEARISH'})
        
        # 98. Price above Cloud
        if not pd.isna(current['Senkou_A']) and not pd.isna(current['Senkou_B']):
            cloud_top = max(current['Senkou_A'], current['Senkou_B'])
            if current['Close'] > cloud_top:
                signals.append({'signal': '🟢 ABOVE CLOUD', 'desc': 'Ichimoku bullish', 'strength': 'BULLISH'})
        
        # 99. High Volatility
        if current['Volatility'] > 50:
            signals.append({'signal': '⚠️ HIGH VOLATILITY', 'desc': f"{current['Volatility']:.0f}% annualized", 'strength': 'CAUTION'})
        
        # 100. Low Volatility
        if current['Volatility'] < 15:
            signals.append({'signal': '⚠️ LOW VOLATILITY', 'desc': f"{current['Volatility']:.0f}% annualized", 'strength': 'CALM'})
        
        self.signals = signals
        
        # Display results
        if signals:
            print(f"\n✅ Detected {len(signals)} Active Signals:\n")
            for i, sig in enumerate(signals, 1):
                print(f"{i}. {sig['signal']}")
                print(f"   📝 {sig['desc']}")
                print(f"   💪 {sig['strength']}")
                print()
        else:
            print("❌ No significant signals detected")
        
        print(f"\n📊 Total Signals Active: {len(signals)}/100")
        
        return signals
    
    def generate_summary(self):
        """Generate comprehensive analysis summary"""
        current = self.data.iloc[-1]
        
        print("\n" + "=" * 80)
        print(f"📊 TECHNICAL ANALYSIS SUMMARY FOR {self.symbol}")
        print("=" * 80)
        
        print(f"\n💰 Current Price: ${current['Close']:.2f}")
        print(f"📅 Date: {current.name.strftime('%Y-%m-%d')}")
        print(f"📈 Change: {current['Price_Change']:.2f}%")
        
        print("\n📈 Moving Averages:")
        print(f"   5 SMA:  ${current['SMA_5']:.2f} ({current['Dist_SMA_10']:.1f}%)")
        print(f"   10 SMA: ${current['SMA_10']:.2f} ({current['Dist_SMA_10']:.1f}%)")
        print(f"   20 SMA: ${current['SMA_20']:.2f} ({current['Dist_SMA_20']:.1f}%)")
        print(f"   50 SMA: ${current['SMA_50']:.2f} ({current['Dist_SMA_50']:.1f}%)")
        if not pd.isna(current['SMA_200']):
            print(f"  200 SMA: ${current['SMA_200']:.2f} ({current['Dist_SMA_200']:.1f}%)")
        
        print("\n📊 Key Oscillators:")
        print(f"   RSI:        {current['RSI']:.1f}")
        print(f"   Stochastic: {current['Stoch_K']:.1f}")
        print(f"   CCI:        {current['CCI']:.1f}")
        print(f"   Williams:   {current['Williams_R']:.1f}")
        print(f"   MFI:        {current['MFI']:.1f}")
        
        print("\n📉 Momentum & Trend:")
        print(f"   MACD:       {current['MACD']:.4f}")
        print(f"   ADX:        {current['ADX']:.1f}")
        print(f"   ATR:        ${current['ATR']:.2f}")
        print(f"   Volatility: {current['Volatility']:.1f}%")
        
        print("\n🎯 Bollinger Bands:")
        print(f"   Upper:  ${current['BB_Upper']:.2f}")
        print(f"   Middle: ${current['BB_Middle']:.2f}")
        print(f"   Lower:  ${current['BB_Lower']:.2f}")
        
        print("\n📊 Volume:")
        print(f"   Current: {current['Volume']:,.0f}")
        print(f"   Avg (20d): {current['Volume_MA_20']:,.0f}")
        
        # Trend assessment
        print("\n🔍 Overall Assessment:")
        bullish = sum(1 for s in self.signals if 'BULLISH' in s['strength'])
        bearish = sum(1 for s in self.signals if 'BEARISH' in s['strength'])
        
        if bullish > bearish * 1.5:
            print("   Bias: 🟢 STRONG BULLISH")
        elif bullish > bearish:
            print("   Bias: 🟢 BULLISH")
        elif bearish > bullish * 1.5:
            print("   Bias: 🔴 STRONG BEARISH")
        elif bearish > bullish:
            print("   Bias: 🔴 BEARISH")
        else:
            print("   Bias: 🟡 NEUTRAL")
        
        print(f"   Bullish Signals: {bullish}")
        print(f"   Bearish Signals: {bearish}")
        print(f"   Total Active: {len(self.signals)}")
        
        print("\n" + "=" * 80)

def main():
    """Main execution"""
    
    SYMBOL = 'RGTI'
    PERIOD = '1y'
    
    print("=" * 80)
    print("🚀 ADVANCED TECHNICAL SCANNER - 100 ALERTS")
    print("=" * 80)
    
    try:
        analyzer = TechnicalAnalyzer(SYMBOL, PERIOD)
        analyzer.fetch_data()
        analyzer.calculate_indicators()
        analyzer.detect_signals()
        analyzer.generate_summary()
        
        print("\n✅ Scan Complete!")
        
        # Save data
        analyzer.data.to_csv(f'{SYMBOL}_analysis.csv')
        print(f"💾 Data saved to {SYMBOL}_analysis.csv")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()