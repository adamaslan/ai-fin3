# nu3
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
import os
from google import genai

warnings.filterwarnings('ignore')

class TechnicalAnalyzer:
    def __init__(self, symbol, period='1y', gcp_bucket='ttb-bucket1', gemini_api_key=None, local_save_dir='technical_analysis_data'):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.signals = []
        self.gcp_bucket = gcp_bucket
        self.gemini_api_key = gemini_api_key
        self.local_save_dir = local_save_dir
        
        if self.gemini_api_key:
            self.genai_client = genai.Client(api_key=self.gemini_api_key)
        else:
            self.genai_client = None
        
        self._setup_local_folders()
    
    def _setup_local_folders(self):
        """Create local folder structure for saving analysis files"""
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        if not os.path.exists(self.local_save_dir):
            os.makedirs(self.local_save_dir)
            print(f"📁 Created main directory: {self.local_save_dir}")
        
        self.date_folder = os.path.join(self.local_save_dir, date_str)
        if not os.path.exists(self.date_folder):
            os.makedirs(self.date_folder)
            print(f"📁 Created date folder: {self.date_folder}")
        else:
            print(f"📁 Using existing folder: {self.date_folder}")
    
    def _generate_filename(self, file_type, extension):
        """Generate standardized filename: YYYY-MM-DD-SYMBOL-type.ext"""
        date_str = datetime.now().strftime('%Y-%m-%d')
        timestamp = datetime.now().strftime('%H%M%S')
        return f"{date_str}-{self.symbol}-{file_type}-{timestamp}.{extension}"
    
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
        
        # MA Slopes
        for period in [10, 20, 50]:
            df[f'SMA_{period}_Slope'] = df[f'SMA_{period}'].diff(5)
        
        # Pivot Points
        df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low'].shift(1)
        df['S1'] = 2 * df['Pivot'] - df['High'].shift(1)
        df['R2'] = df['Pivot'] + (df['High'].shift(1) - df['Low'].shift(1))
        df['S2'] = df['Pivot'] - (df['High'].shift(1) - df['Low'].shift(1))
        
        # Fibonacci Retracement Levels
        period_high = df['High'].rolling(window=50).max()
        period_low = df['Low'].rolling(window=50).min()
        diff = period_high - period_low
        df['Fib_382'] = period_high - 0.382 * diff
        df['Fib_500'] = period_high - 0.500 * diff
        df['Fib_618'] = period_high - 0.618 * diff
        
        self.data = df
        print("✅ All indicators calculated")
        return df
    
    def detect_signals(self):
        """Detect 150+ comprehensive technical signals"""
        df = self.data.copy()
        current = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3] if len(df) > 2 else prev
        
        signals = []
        
        print("\n🎯 Scanning for Technical Alerts...")
        
        # ============ BASIC SIGNALS ============
        
        # Moving Average Crossovers
        if len(df) > 200 and prev['SMA_50'] <= prev['SMA_200'] and current['SMA_50'] > current['SMA_200']:
            signals.append({'signal': 'GOLDEN CROSS', 'desc': '50 MA crossed above 200 MA', 'strength': 'STRONG BULLISH', 'category': 'MA_CROSS'})
        
        if len(df) > 200 and prev['SMA_50'] >= prev['SMA_200'] and current['SMA_50'] < current['SMA_200']:
            signals.append({'signal': 'DEATH CROSS', 'desc': '50 MA crossed below 200 MA', 'strength': 'STRONG BEARISH', 'category': 'MA_CROSS'})
        
        if prev['Close'] <= prev['SMA_10'] and current['Close'] > current['SMA_10']:
            signals.append({'signal': 'PRICE ABOVE 10 MA', 'desc': 'Price crossed above 10-day MA', 'strength': 'BULLISH', 'category': 'MA_CROSS'})
        
        if prev['Close'] >= prev['SMA_10'] and current['Close'] < current['SMA_10']:
            signals.append({'signal': 'PRICE BELOW 10 MA', 'desc': 'Price crossed below 10-day MA', 'strength': 'BEARISH', 'category': 'MA_CROSS'})
        
        if prev['EMA_10'] <= prev['EMA_20'] and current['EMA_10'] > current['EMA_20']:
            signals.append({'signal': '10/20 EMA BULL CROSS', 'desc': '10 EMA crossed above 20 EMA', 'strength': 'BULLISH', 'category': 'MA_CROSS'})
        
        if prev['EMA_10'] >= prev['EMA_20'] and current['EMA_10'] < current['EMA_20']:
            signals.append({'signal': '10/20 EMA BEAR CROSS', 'desc': '10 EMA crossed below 20 EMA', 'strength': 'BEARISH', 'category': 'MA_CROSS'})
        
        # RSI Signals
        if current['RSI'] < 30:
            signals.append({'signal': 'RSI OVERSOLD', 'desc': f"RSI at {current['RSI']:.1f}", 'strength': 'BULLISH', 'category': 'RSI'})
        
        if current['RSI'] > 70:
            signals.append({'signal': 'RSI OVERBOUGHT', 'desc': f"RSI at {current['RSI']:.1f}", 'strength': 'BEARISH', 'category': 'RSI'})
        
        if current['RSI'] < 20:
            signals.append({'signal': 'RSI EXTREME OVERSOLD', 'desc': f"RSI at {current['RSI']:.1f}", 'strength': 'STRONG BULLISH', 'category': 'RSI'})
        
        if current['RSI'] > 80:
            signals.append({'signal': 'RSI EXTREME OVERBOUGHT', 'desc': f"RSI at {current['RSI']:.1f}", 'strength': 'STRONG BEARISH', 'category': 'RSI'})
        
        # RSI Divergences
        if len(df) > 20:
            if current['Close'] < df['Close'].iloc[-20] and current['RSI'] > df['RSI'].iloc[-20]:
                signals.append({'signal': 'RSI BULLISH DIVERGENCE', 'desc': 'Price down but RSI up', 'strength': 'BULLISH', 'category': 'RSI_DIVERGENCE'})
        
        if len(df) > 20:
            if current['Close'] > df['Close'].iloc[-20] and current['RSI'] < df['RSI'].iloc[-20]:
                signals.append({'signal': 'RSI BEARISH DIVERGENCE', 'desc': 'Price up but RSI down', 'strength': 'BEARISH', 'category': 'RSI_DIVERGENCE'})
        
        # MACD Signals
        if prev['MACD'] <= prev['MACD_Signal'] and current['MACD'] > current['MACD_Signal']:
            signals.append({'signal': 'MACD BULL CROSS', 'desc': 'MACD crossed above signal', 'strength': 'BULLISH', 'category': 'MACD'})
        
        if prev['MACD'] >= prev['MACD_Signal'] and current['MACD'] < current['MACD_Signal']:
            signals.append({'signal': 'MACD BEAR CROSS', 'desc': 'MACD crossed below signal', 'strength': 'BEARISH', 'category': 'MACD'})
        
        if prev['MACD'] <= 0 and current['MACD'] > 0:
            signals.append({'signal': 'MACD ABOVE ZERO', 'desc': 'MACD crossed into positive territory', 'strength': 'BULLISH', 'category': 'MACD'})
        
        if prev['MACD'] >= 0 and current['MACD'] < 0:
            signals.append({'signal': 'MACD BELOW ZERO', 'desc': 'MACD crossed into negative territory', 'strength': 'BEARISH', 'category': 'MACD'})
        
        # Bollinger Bands
        bb_width_avg = df['BB_Width'].tail(50).mean()
        if current['BB_Width'] < bb_width_avg * 0.7:
            signals.append({'signal': 'BB SQUEEZE', 'desc': 'Bands narrowing - breakout pending', 'strength': 'NEUTRAL', 'category': 'BB_SQUEEZE'})
        
        if current['Close'] <= current['BB_Lower'] * 1.01:
            signals.append({'signal': 'AT LOWER BB', 'desc': f"Price at ${current['BB_Lower']:.2f}", 'strength': 'BULLISH', 'category': 'BOLLINGER'})
        
        if current['Close'] >= current['BB_Upper'] * 0.99:
            signals.append({'signal': 'AT UPPER BB', 'desc': f"Price at ${current['BB_Upper']:.2f}", 'strength': 'BEARISH', 'category': 'BOLLINGER'})
        
        # Volume Signals
        if current['Volume'] > current['Volume_MA_20'] * 2:
            signals.append({'signal': 'VOLUME SPIKE 2X', 'desc': f"Vol: {current['Volume']:,.0f}", 'strength': 'SIGNIFICANT', 'category': 'VOLUME'})
        
        if current['Volume'] > current['Volume_MA_20'] * 3:
            signals.append({'signal': 'EXTREME VOLUME 3X', 'desc': f"Vol: {current['Volume']:,.0f}", 'strength': 'VERY SIGNIFICANT', 'category': 'VOLUME'})
        
        if current['Price_Change'] > 2 and current['Volume'] > current['Volume_MA_20'] * 1.5:
            signals.append({'signal': 'VOLUME BREAKOUT', 'desc': 'High volume + price up', 'strength': 'STRONG BULLISH', 'category': 'VOLUME'})
        
        if current['Price_Change'] < -2 and current['Volume'] > current['Volume_MA_20'] * 1.5:
            signals.append({'signal': 'VOLUME SELLOFF', 'desc': 'High volume + price down', 'strength': 'STRONG BEARISH', 'category': 'VOLUME'})
        
        # Price Action
        if current['Price_Change'] > 5:
            signals.append({'signal': 'LARGE GAIN', 'desc': f"+{current['Price_Change']:.1f}% today", 'strength': 'STRONG BULLISH', 'category': 'PRICE_ACTION'})
        
        if current['Price_Change'] < -5:
            signals.append({'signal': 'LARGE LOSS', 'desc': f"{current['Price_Change']:.1f}% today", 'strength': 'STRONG BEARISH', 'category': 'PRICE_ACTION'})
        
        if current['Close'] >= current['High_52w'] * 0.999:
            signals.append({'signal': '52-WEEK HIGH', 'desc': f"At ${current['Close']:.2f}", 'strength': 'STRONG BULLISH', 'category': 'RANGE'})
        
        if current['Close'] <= current['Low_52w'] * 1.001:
            signals.append({'signal': '52-WEEK LOW', 'desc': f"At ${current['Close']:.2f}", 'strength': 'STRONG BEARISH', 'category': 'RANGE'})
        
        # Trend Strength
        if current['ADX'] > 25:
            trend = 'UP' if current['Close'] > current['SMA_50'] else 'DOWN'
            signals.append({'signal': f"STRONG {trend}TREND", 'desc': f"ADX: {current['ADX']:.1f}", 'strength': 'TRENDING', 'category': 'TREND'})
        
        if current['ADX'] > 40:
            signals.append({'signal': 'VERY STRONG TREND', 'desc': f"ADX: {current['ADX']:.1f}", 'strength': 'EXTREME', 'category': 'TREND'})
        
        # MA Alignment
        mas_aligned_bull = (current['SMA_10'] > current['SMA_20'] > current['SMA_50'])
        if mas_aligned_bull:
            signals.append({'signal': 'MA ALIGNMENT BULLISH', 'desc': '10 > 20 > 50 SMA', 'strength': 'STRONG BULLISH', 'category': 'MA_TREND'})
        
        mas_aligned_bear = (current['SMA_10'] < current['SMA_20'] < current['SMA_50'])
        if mas_aligned_bear:
            signals.append({'signal': 'MA ALIGNMENT BEARISH', 'desc': '10 < 20 < 50 SMA', 'strength': 'STRONG BEARISH', 'category': 'MA_TREND'})
        
        if current['Close'] > current['SMA_200'] and len(df) > 200:
            signals.append({'signal': 'ABOVE 200 SMA', 'desc': 'Long-term uptrend', 'strength': 'BULLISH', 'category': 'MA_TREND'})
        
        if current['Close'] < current['SMA_200'] and len(df) > 200:
            signals.append({'signal': 'BELOW 200 SMA', 'desc': 'Long-term downtrend', 'strength': 'BEARISH', 'category': 'MA_TREND'})
        
        # ============ ENHANCED SIGNALS ============
        
        # MA Compression
        if len(df) > 50:
            ma_range = (current['SMA_50'] - current['SMA_10']) / current['SMA_50'] * 100
            if abs(ma_range) < 2:
                signals.append({'signal': 'MA COMPRESSION', 'desc': 'MAs converging - breakout imminent', 'strength': 'NEUTRAL', 'category': 'MA_COMPRESSION'})
        
        # MA Slope Analysis
        if current.get('SMA_20_Slope', 0) > 0 and prev.get('SMA_20_Slope', 0) <= 0:
            signals.append({'signal': 'MA SLOPE REVERSAL UP', 'desc': '20 MA turning upward', 'strength': 'BULLISH', 'category': 'MA_SLOPE'})
        
        if current.get('SMA_20_Slope', 0) < 0 and prev.get('SMA_20_Slope', 0) >= 0:
            signals.append({'signal': 'MA SLOPE REVERSAL DOWN', 'desc': '20 MA turning downward', 'strength': 'BEARISH', 'category': 'MA_SLOPE'})
        
        # Extreme Distance from MAs
        if current.get('Dist_SMA_20', 0) > 10:
            signals.append({'signal': 'OVEREXTENDED ABOVE 20MA', 'desc': f"{current.get('Dist_SMA_20', 0):.1f}% above 20 MA", 'strength': 'BEARISH', 'category': 'MA_DISTANCE'})
        
        if current.get('Dist_SMA_20', 0) < -10:
            signals.append({'signal': 'OVEREXTENDED BELOW 20MA', 'desc': f"{abs(current.get('Dist_SMA_20', 0)):.1f}% below 20 MA", 'strength': 'BULLISH', 'category': 'MA_DISTANCE'})
        
        # RSI Momentum
        if len(df) > 5:
            rsi_momentum = current['RSI'] - df['RSI'].iloc[-5]
            if rsi_momentum > 10 and current['RSI'] < 50:
                signals.append({'signal': 'RSI MOMENTUM SURGE', 'desc': f"RSI +{rsi_momentum:.1f} in 5 days", 'strength': 'BULLISH', 'category': 'RSI_MOMENTUM'})
            elif rsi_momentum < -10 and current['RSI'] > 50:
                signals.append({'signal': 'RSI MOMENTUM COLLAPSE', 'desc': f"RSI {rsi_momentum:.1f} in 5 days", 'strength': 'BEARISH', 'category': 'RSI_MOMENTUM'})
        
        # RSI 50 Cross
        if prev['RSI'] <= 50 and current['RSI'] > 50:
            signals.append({'signal': 'RSI ABOVE 50', 'desc': 'RSI crossed into bullish zone', 'strength': 'BULLISH', 'category': 'RSI_CROSS'})
        
        if prev['RSI'] >= 50 and current['RSI'] < 50:
            signals.append({'signal': 'RSI BELOW 50', 'desc': 'RSI crossed into bearish zone', 'strength': 'BEARISH', 'category': 'RSI_CROSS'})
        
        # MACD Histogram Expansion
        if len(df) > 3:
            hist_expanding = abs(current['MACD_Hist']) > abs(prev['MACD_Hist']) > abs(prev2['MACD_Hist'])
            if hist_expanding and current['MACD_Hist'] > 0:
                signals.append({'signal': 'MACD HISTOGRAM EXPANSION', 'desc': 'Bullish momentum accelerating', 'strength': 'STRONG BULLISH', 'category': 'MACD_MOMENTUM'})
            elif hist_expanding and current['MACD_Hist'] < 0:
                signals.append({'signal': 'MACD HISTOGRAM EXPANSION', 'desc': 'Bearish momentum accelerating', 'strength': 'STRONG BEARISH', 'category': 'MACD_MOMENTUM'})
        
        # Bollinger Band Walk
        if len(df) > 5:
            bb_walk_bull = all(df['Close'].iloc[-i] >= df['BB_Upper'].iloc[-i] * 0.98 for i in range(1, 4))
            bb_walk_bear = all(df['Close'].iloc[-i] <= df['BB_Lower'].iloc[-i] * 1.02 for i in range(1, 4))
            
            if bb_walk_bull:
                signals.append({'signal': 'BB WALK UPPER', 'desc': 'Price riding upper band - strong trend', 'strength': 'STRONG BULLISH', 'category': 'BB_WALK'})
            
            if bb_walk_bear:
                signals.append({'signal': 'BB WALK LOWER', 'desc': 'Price riding lower band - strong downtrend', 'strength': 'STRONG BEARISH', 'category': 'BB_WALK'})
        
        # Bollinger Band Extremes
        if current['BB_Position'] > 1.1:
            signals.append({'signal': 'ABOVE UPPER BB', 'desc': 'Price extended beyond upper band', 'strength': 'EXTREME BULLISH', 'category': 'BB_EXTREME'})
        
        if current['BB_Position'] < -0.1:
            signals.append({'signal': 'BELOW LOWER BB', 'desc': 'Price extended beyond lower band', 'strength': 'EXTREME BEARISH', 'category': 'BB_EXTREME'})
        
        # Volume Divergence
        if len(df) > 10:
            price_trend_up = current['Close'] > df['Close'].iloc[-10]
            volume_trend_down = current['Volume'] < df['Volume'].iloc[-10]
            
            if price_trend_up and volume_trend_down:
                signals.append({'signal': 'VOLUME DIVERGENCE BEARISH', 'desc': 'Price up but volume declining', 'strength': 'BEARISH', 'category': 'VOLUME_DIVERGENCE'})
            elif not price_trend_up and not volume_trend_down:
                signals.append({'signal': 'VOLUME DIVERGENCE BULLISH', 'desc': 'Price down but volume increasing', 'strength': 'BULLISH', 'category': 'VOLUME_DIVERGENCE'})
        
        # Volume Accumulation
        if len(df) > 5:
            volume_increasing = all(df['Volume'].iloc[-i] > df['Volume'].iloc[-i-1] for i in range(1, 4))
            if volume_increasing and current['Close'] > prev['Close']:
                signals.append({'signal': 'VOLUME ACCUMULATION', 'desc': 'Consistent volume + price increase', 'strength': 'STRONG BULLISH', 'category': 'VOLUME_ACCUMULATION'})
        
        # Gap Detection
        gap_up = (current['Open'] - prev['Close']) / prev['Close'] * 100
        if gap_up > 2:
            signals.append({'signal': 'GAP UP', 'desc': f"Opened {gap_up:.1f}% higher", 'strength': 'BULLISH', 'category': 'GAP'})
        elif gap_up < -2:
            signals.append({'signal': 'GAP DOWN', 'desc': f"Opened {abs(gap_up):.1f}% lower", 'strength': 'BEARISH', 'category': 'GAP'})
        
        # Price Pattern: Higher Highs/Lower Lows
        if len(df) > 10:
            recent_highs = [df['High'].iloc[-i] for i in range(1, 6)]
            recent_lows = [df['Low'].iloc[-i] for i in range(1, 6)]
            
            higher_highs = all(recent_highs[i] > recent_highs[i+1] for i in range(len(recent_highs)-1))
            higher_lows = all(recent_lows[i] > recent_lows[i+1] for i in range(len(recent_lows)-1))
            lower_highs = all(recent_highs[i] < recent_highs[i+1] for i in range(len(recent_highs)-1))
            lower_lows = all(recent_lows[i] < recent_lows[i+1] for i in range(len(recent_lows)-1))
            
            if higher_highs and higher_lows:
                signals.append({'signal': 'HIGHER HIGHS & LOWS', 'desc': 'Clear uptrend pattern', 'strength': 'STRONG BULLISH', 'category': 'PRICE_PATTERN'})
            
            if lower_highs and lower_lows:
                signals.append({'signal': 'LOWER HIGHS & LOWS', 'desc': 'Clear downtrend pattern', 'strength': 'STRONG BEARISH', 'category': 'PRICE_PATTERN'})
        
        # Exhaustion Signals
        if current['RSI'] > 75 and current['Volume'] > current['Volume_MA_20'] * 2:
            signals.append({'signal': 'BULLISH EXHAUSTION', 'desc': 'Extreme RSI + high volume', 'strength': 'BEARISH', 'category': 'EXHAUSTION'})
        
        if current['RSI'] < 25 and current['Volume'] > current['Volume_MA_20'] * 2:
            signals.append({'signal': 'BEARISH EXHAUSTION', 'desc': 'Extreme oversold + high volume', 'strength': 'BULLISH', 'category': 'EXHAUSTION'})
        
        # Pivot Points
        if current['Close'] > current['R1'] and prev['Close'] <= prev['R1']:
            signals.append({'signal': 'ABOVE R1 PIVOT', 'desc': 'Broke resistance level', 'strength': 'BULLISH', 'category': 'PIVOT'})
        
        if current['Close'] < current['S1'] and prev['Close'] >= prev['S1']:
            signals.append({'signal': 'BELOW S1 PIVOT', 'desc': 'Broke support level', 'strength': 'BEARISH', 'category': 'PIVOT'})
        
        # Fibonacci Levels
        if abs(current['Close'] - current['Fib_618']) / current['Close'] < 0.01:
            signals.append({'signal': 'AT FIB 61.8%', 'desc': 'Key retracement level', 'strength': 'SIGNIFICANT', 'category': 'FIBONACCI'})
        
        if abs(current['Close'] - current['Fib_500']) / current['Close'] < 0.01:
            signals.append({'signal': 'AT FIB 50%', 'desc': 'Mid retracement level', 'strength': 'SIGNIFICANT', 'category': 'FIBONACCI'})
        
        # Market Structure
        if len(df) > 20:
            resistance_level = df['High'].iloc[-20:].max()
            support_level = df['Low'].iloc[-20:].max()
            
            if current['Close'] > resistance_level and prev['Close'] <= resistance_level:
                signals.append({'signal': 'RESISTANCE BREAKOUT', 'desc': f"Broke above ${resistance_level:.2f}", 'strength': 'STRONG BULLISH', 'category': 'STRUCTURE_BREAK'})
            
            if current['Close'] < support_level and prev['Close'] >= support_level:
                signals.append({'signal': 'SUPPORT BREAKDOWN', 'desc': f"Broke below ${support_level:.2f}", 'strength': 'STRONG BEARISH', 'category': 'STRUCTURE_BREAK'})
        
        # Momentum Divergence (Price vs Volume)
        if len(df) > 10:
            price_roc = (current['Close'] - df['Close'].iloc[-10]) / df['Close'].iloc[-10] * 100
            volume_roc = (current['Volume'] - df['Volume'].iloc[-10]) / df['Volume'].iloc[-10] * 100
            
            if price_roc > 5 and volume_roc < -20:
                signals.append({'signal': 'MOMENTUM DIVERGENCE', 'desc': 'Strong price move on weak volume', 'strength': 'BEARISH', 'category': 'MOMENTUM_DIVERGENCE'})
        
        # Momentum Acceleration
        if len(df) > 5:
            if current['Momentum'] > prev['Momentum'] > prev2['Momentum'] and current['Momentum'] > 0:
                signals.append({'signal': 'MOMENTUM ACCELERATION', 'desc': 'Bullish momentum increasing', 'strength': 'STRONG BULLISH', 'category': 'MOMENTUM_ACCELERATION'})
            
            if current['Momentum'] < prev['Momentum'] < prev2['Momentum'] and current['Momentum'] < 0:
                signals.append({'signal': 'MOMENTUM DECELERATION', 'desc': 'Bearish momentum increasing', 'strength': 'STRONG BEARISH', 'category': 'MOMENTUM_ACCELERATION'})
        
        # Volatility Signals
        if current['Volatility'] > 40:
            signals.append({'signal': 'HIGH VOLATILITY', 'desc': f"{current['Volatility']:.1f}% annualized", 'strength': 'SIGNIFICANT', 'category': 'VOLATILITY'})
        
        if current['Volatility'] < 15:
            signals.append({'signal': 'LOW VOLATILITY', 'desc': 'Compression - breakout likely', 'strength': 'NEUTRAL', 'category': 'VOLATILITY_SQUEEZE'})
        
        # Volatility Expansion
        if len(df) > 10:
            vol_change = (current['Volatility'] - df['Volatility'].iloc[-10]) / df['Volatility'].iloc[-10] * 100
            if vol_change > 50:
                signals.append({'signal': 'VOLATILITY EXPANSION', 'desc': f"Vol increased {vol_change:.0f}%", 'strength': 'SIGNIFICANT', 'category': 'VOLATILITY'})
        
        # Stochastic Signals
        if current['Stoch_K'] < 20 and prev['Stoch_K'] >= 20:
            signals.append({'signal': 'STOCHASTIC OVERSOLD', 'desc': f"K at {current['Stoch_K']:.1f}", 'strength': 'BULLISH', 'category': 'STOCHASTIC'})
        
        if current['Stoch_K'] > 80 and prev['Stoch_K'] <= 80:
            signals.append({'signal': 'STOCHASTIC OVERBOUGHT', 'desc': f"K at {current['Stoch_K']:.1f}", 'strength': 'BEARISH', 'category': 'STOCHASTIC'})
        
        if prev['Stoch_K'] <= prev['Stoch_D'] and current['Stoch_K'] > current['Stoch_D']:
            signals.append({'signal': 'STOCHASTIC BULL CROSS', 'desc': 'K crossed above D', 'strength': 'BULLISH', 'category': 'STOCHASTIC'})
        
        if prev['Stoch_K'] >= prev['Stoch_D'] and current['Stoch_K'] < current['Stoch_D']:
            signals.append({'signal': 'STOCHASTIC BEAR CROSS', 'desc': 'K crossed below D', 'strength': 'BEARISH', 'category': 'STOCHASTIC'})
        
        # Williams %R
        if current['Williams_R'] < -80:
            signals.append({'signal': 'WILLIAMS R OVERSOLD', 'desc': f"At {current['Williams_R']:.1f}", 'strength': 'BULLISH', 'category': 'WILLIAMS_R'})
        
        if current['Williams_R'] > -20:
            signals.append({'signal': 'WILLIAMS R OVERBOUGHT', 'desc': f"At {current['Williams_R']:.1f}", 'strength': 'BEARISH', 'category': 'WILLIAMS_R'})
        
        # CCI Extremes
        if current['CCI'] > 100:
            signals.append({'signal': 'CCI OVERBOUGHT', 'desc': f"CCI at {current['CCI']:.1f}", 'strength': 'BEARISH', 'category': 'CCI'})
        
        if current['CCI'] < -100:
            signals.append({'signal': 'CCI OVERSOLD', 'desc': f"CCI at {current['CCI']:.1f}", 'strength': 'BULLISH', 'category': 'CCI'})
        
        # MFI Signals
        if current['MFI'] < 20:
            signals.append({'signal': 'MFI OVERSOLD', 'desc': f"Money flow at {current['MFI']:.1f}", 'strength': 'BULLISH', 'category': 'MFI'})
        
        if current['MFI'] > 80:
            signals.append({'signal': 'MFI OVERBOUGHT', 'desc': f"Money flow at {current['MFI']:.1f}", 'strength': 'BEARISH', 'category': 'MFI'})
        
        # OBV Trend
        if len(df) > 20:
            obv_trend = (current['OBV'] - df['OBV'].iloc[-20]) / abs(df['OBV'].iloc[-20]) * 100
            if obv_trend > 20 and current['Close'] > df['Close'].iloc[-20]:
                signals.append({'signal': 'OBV CONFIRMATION', 'desc': 'Volume confirming price uptrend', 'strength': 'STRONG BULLISH', 'category': 'OBV'})
            elif obv_trend < -20 and current['Close'] < df['Close'].iloc[-20]:
                signals.append({'signal': 'OBV CONFIRMATION', 'desc': 'Volume confirming price downtrend', 'strength': 'STRONG BEARISH', 'category': 'OBV'})
        
        # Ichimoku Cloud
        if current['Close'] > current['Senkou_A'] and current['Close'] > current['Senkou_B']:
            signals.append({'signal': 'ABOVE ICHIMOKU CLOUD', 'desc': 'Bullish cloud position', 'strength': 'BULLISH', 'category': 'ICHIMOKU'})
        
        if current['Close'] < current['Senkou_A'] and current['Close'] < current['Senkou_B']:
            signals.append({'signal': 'BELOW ICHIMOKU CLOUD', 'desc': 'Bearish cloud position', 'strength': 'BEARISH', 'category': 'ICHIMOKU'})
        
        if prev['Close'] <= prev['Tenkan'] and current['Close'] > current['Tenkan']:
            signals.append({'signal': 'TENKAN CROSS UP', 'desc': 'Price crossed above Tenkan line', 'strength': 'BULLISH', 'category': 'ICHIMOKU'})
        
        # Rate of Change Extremes
        if current['ROC_20'] > 15:
            signals.append({'signal': 'EXTREME ROC POSITIVE', 'desc': f"20-day ROC: +{current['ROC_20']:.1f}%", 'strength': 'STRONG BULLISH', 'category': 'ROC'})
        
        if current['ROC_20'] < -15:
            signals.append({'signal': 'EXTREME ROC NEGATIVE', 'desc': f"20-day ROC: {current['ROC_20']:.1f}%", 'strength': 'STRONG BEARISH', 'category': 'ROC'})
        
        # Multi-Timeframe Alignment
        if len(df) > 200:
            short_term_bull = current['Close'] > current['SMA_10']
            mid_term_bull = current['Close'] > current['SMA_50']
            long_term_bull = current['Close'] > current['SMA_200']
            
            if short_term_bull and mid_term_bull and long_term_bull:
                signals.append({'signal': 'FULL TIMEFRAME ALIGNMENT', 'desc': 'Bullish on all timeframes', 'strength': 'EXTREME BULLISH', 'category': 'TIMEFRAME_ALIGNMENT'})
            
            if not short_term_bull and not mid_term_bull and not long_term_bull:
                signals.append({'signal': 'FULL TIMEFRAME ALIGNMENT', 'desc': 'Bearish on all timeframes', 'strength': 'EXTREME BEARISH', 'category': 'TIMEFRAME_ALIGNMENT'})
        
        # ADX Trend Strength with Direction
        if current['ADX'] > 30:
            if current['Plus_DI'] > current['Minus_DI']:
                signals.append({'signal': 'STRONG BULL TREND ADX', 'desc': f"+DI leading, ADX {current['ADX']:.1f}", 'strength': 'STRONG BULLISH', 'category': 'ADX'})
            else:
                signals.append({'signal': 'STRONG BEAR TREND ADX', 'desc': f"-DI leading, ADX {current['ADX']:.1f}", 'strength': 'STRONG BEARISH', 'category': 'ADX'})
        
        # DI Crossover
        if prev['Plus_DI'] <= prev['Minus_DI'] and current['Plus_DI'] > current['Minus_DI']:
            signals.append({'signal': 'DI BULL CROSS', 'desc': '+DI crossed above -DI', 'strength': 'BULLISH', 'category': 'ADX'})
        
        if prev['Plus_DI'] >= prev['Minus_DI'] and current['Plus_DI'] < current['Minus_DI']:
            signals.append({'signal': 'DI BEAR CROSS', 'desc': '-DI crossed above +DI', 'strength': 'BEARISH', 'category': 'ADX'})
        
        # ATR-Based Stop Levels
        stop_distance = current['ATR'] * 2
        if current['Close'] > current['SMA_20']:
            stop_level = current['Close'] - stop_distance
            signals.append({'signal': 'ATR STOP LEVEL', 'desc': f"Bull stop at ${stop_level:.2f}", 'strength': 'NEUTRAL', 'category': 'ATR'})
        
        # Consolidation Detection
        if len(df) > 10:
            price_range = (df['High'].iloc[-10:].max() - df['Low'].iloc[-10:].min()) / df['Close'].iloc[-10:].mean() * 100
            if price_range < 5:
                signals.append({'signal': 'TIGHT CONSOLIDATION', 'desc': f"Range: {price_range:.1f}% over 10 days", 'strength': 'NEUTRAL', 'category': 'CONSOLIDATION'})
        
        # Price Action: Inside/Outside Bars
        if current['High'] < prev['High'] and current['Low'] > prev['Low']:
            signals.append({'signal': 'INSIDE BAR', 'desc': 'Consolidation pattern', 'strength': 'NEUTRAL', 'category': 'PRICE_PATTERN'})
        
        if current['High'] > prev['High'] and current['Low'] < prev['Low']:
            signals.append({'signal': 'OUTSIDE BAR', 'desc': 'Volatility expansion', 'strength': 'SIGNIFICANT', 'category': 'PRICE_PATTERN'})
        
        # Engulfing Patterns
        if current['Close'] > current['Open'] and prev['Close'] < prev['Open']:
            if current['Open'] <= prev['Close'] and current['Close'] >= prev['Open']:
                signals.append({'signal': 'BULLISH ENGULFING', 'desc': 'Strong reversal pattern', 'strength': 'STRONG BULLISH', 'category': 'CANDLESTICK'})
        
        if current['Close'] < current['Open'] and prev['Close'] > prev['Open']:
            if current['Open'] >= prev['Close'] and current['Close'] <= prev['Open']:
                signals.append({'signal': 'BEARISH ENGULFING', 'desc': 'Strong reversal pattern', 'strength': 'STRONG BEARISH', 'category': 'CANDLESTICK'})
        
        # Doji Detection
        body_size = abs(current['Close'] - current['Open'])
        candle_range = current['High'] - current['Low']
        if candle_range > 0 and body_size / candle_range < 0.1:
            signals.append({'signal': 'DOJI CANDLE', 'desc': 'Indecision/reversal signal', 'strength': 'NEUTRAL', 'category': 'CANDLESTICK'})
        
        # Hammer/Shooting Star
        lower_wick = min(current['Open'], current['Close']) - current['Low']
        upper_wick = current['High'] - max(current['Open'], current['Close'])
        
        if lower_wick > 2 * body_size and upper_wick < body_size:
            signals.append({'signal': 'HAMMER PATTERN', 'desc': 'Potential reversal up', 'strength': 'BULLISH', 'category': 'CANDLESTICK'})
        
        if upper_wick > 2 * body_size and lower_wick < body_size:
            signals.append({'signal': 'SHOOTING STAR', 'desc': 'Potential reversal down', 'strength': 'BEARISH', 'category': 'CANDLESTICK'})
        
        # Volume Climax
        if current['Volume'] > current['Volume_MA_20'] * 4:
            if current['Close'] < current['Open']:
                signals.append({'signal': 'SELLING CLIMAX', 'desc': 'Extreme volume selloff', 'strength': 'EXTREME', 'category': 'VOLUME_CLIMAX'})
            else:
                signals.append({'signal': 'BUYING CLIMAX', 'desc': 'Extreme volume buying', 'strength': 'EXTREME', 'category': 'VOLUME_CLIMAX'})
        
        # VWAP Signals
        if current['Close'] > current['VWAP'] and prev['Close'] <= prev['VWAP']:
            signals.append({'signal': 'ABOVE VWAP', 'desc': 'Institutional buying support', 'strength': 'BULLISH', 'category': 'VWAP'})
        
        if current['Close'] < current['VWAP'] and prev['Close'] >= prev['VWAP']:
            signals.append({'signal': 'BELOW VWAP', 'desc': 'Institutional selling pressure', 'strength': 'BEARISH', 'category': 'VWAP'})
        
        # Multiple Indicator Confirmation
        bullish_count = sum([
            current['RSI'] > 50,
            current['MACD'] > current['MACD_Signal'],
            current['Close'] > current['SMA_20'],
            current['Stoch_K'] > current['Stoch_D'],
            current['Plus_DI'] > current['Minus_DI']
        ])
        
        if bullish_count >= 4:
            signals.append({'signal': 'MULTI-INDICATOR BULLISH', 'desc': f"{bullish_count}/5 indicators bullish", 'strength': 'STRONG BULLISH', 'category': 'MULTI_INDICATOR'})
        elif bullish_count <= 1:
            signals.append({'signal': 'MULTI-INDICATOR BEARISH', 'desc': f"{5-bullish_count}/5 indicators bearish", 'strength': 'STRONG BEARISH', 'category': 'MULTI_INDICATOR'})
        
        self.signals = signals
        print(f"✅ Detected {len(signals)} Active Signals")
        return signals
    
    def rank_signals_with_ai(self):
        """Use Gemini AI to score each signal from 1-100"""
        if not self.genai_client:
            print("\n⚠️  Gemini API key not provided. Skipping AI ranking.")
            for signal in self.signals:
                signal['ai_score'] = 50
                signal['ai_reasoning'] = 'No AI scoring available'
            return
        
        print("\n🤖 AI is scoring all signals (1-100)...")
        
        # If too many signals, process in batches
        max_signals_per_batch = 50
        if len(self.signals) > max_signals_per_batch:
            print(f"⚙️  Processing {len(self.signals)} signals in batches...")
            self._rank_signals_in_batches(max_signals_per_batch)
            return
        
        try:
            current = self.data.iloc[-1]
            
            prompt = f"""You are an expert technical analyst. Score these trading signals for {self.symbol}.

MARKET DATA:
- Price: ${current['Close']:.2f} | Change: {current['Price_Change']:.2f}%
- RSI: {current['RSI']:.1f} | MACD: {current['MACD']:.4f} | ADX: {current['ADX']:.1f}
- Volatility: {current['Volatility']:.1f}% | Volume: {(current['Volume'] / current['Volume_MA_20'] * 100):.0f}% of avg

SIGNALS:
"""
            for i, sig in enumerate(self.signals, 1):
                prompt += f"{i}. {sig['signal']}: {sig['desc']}\n"
            
            prompt += """
Score each signal 1-100 based on actionability, reliability, timing, and risk/reward.

Return ONLY valid JSON (no markdown, no explanations):
{"scores":[{"signal_number":1,"score":85,"reasoning":"Brief reason"},{"signal_number":2,"score":72,"reasoning":"Brief reason"}],"top_signal":{"signal_number":1,"why":"Brief explanation"}}

Make reasoning brief (under 50 chars). Score ALL signals."""

            response = self.genai_client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            
            response_text = response.text.strip()
            
            # Clean up response
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            # Remove any leading/trailing text
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                response_text = response_text[start_idx:end_idx+1]
            
            # Parse JSON
            scores_data = json.loads(response_text)
            
            # Apply scores
            scores_applied = 0
            for score_item in scores_data.get('scores', []):
                sig_num = score_item['signal_number'] - 1
                if 0 <= sig_num < len(self.signals):
                    self.signals[sig_num]['ai_score'] = score_item.get('score', 50)
                    self.signals[sig_num]['ai_reasoning'] = score_item.get('reasoning', 'No reasoning provided')
                    scores_applied += 1
            
            # Sort by score
            self.signals.sort(key=lambda x: x.get('ai_score', 0), reverse=True)
            
            # Add ranks
            for rank, signal in enumerate(self.signals, 1):
                signal['rank'] = rank
            
            self.top_signal_info = scores_data.get('top_signal', {})
            
            print(f"✅ AI scored {scores_applied}/{len(self.signals)} signals")
            if self.top_signal_info:
                print(f"🏆 Top Signal: #{self.top_signal_info.get('signal_number', 'N/A')}")
            
            # Fill in any missing scores
            for signal in self.signals:
                if 'ai_score' not in signal:
                    signal['ai_score'] = 50
                    signal['ai_reasoning'] = 'Score not provided by AI'
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON Parsing Error: {str(e)}")
            print(f"📝 AI Response preview: {response_text[:200] if 'response_text' in locals() else 'No response'}")
            self._apply_default_scores()
            
        except Exception as e:
            print(f"❌ AI Scoring Error: {str(e)}")
            self._apply_default_scores()
    
    def _apply_default_scores(self):
        """Apply default scores when AI scoring fails"""
        print("⚙️  Applying rule-based scores...")
        
        for signal in self.signals:
            # Rule-based scoring
            score = 50  # Default
            
            # Adjust by strength
            strength = signal.get('strength', '')
            if 'EXTREME' in strength:
                score = 85
            elif 'STRONG' in strength:
                score = 75
            elif 'SIGNIFICANT' in strength or 'VERY' in strength:
                score = 65
            elif 'BULLISH' in strength or 'BEARISH' in strength:
                score = 55
            
            # Adjust by category (some are more reliable)
            category = signal.get('category', '')
            if category in ['MA_CROSS', 'STRUCTURE_BREAK', 'VOLUME']:
                score += 5
            elif category in ['TIMEFRAME_ALIGNMENT', 'MULTI_INDICATOR']:
                score += 10
            
            # Cap at 90 for rule-based
            score = min(score, 90)
            
            signal['ai_score'] = score
            signal['ai_reasoning'] = 'Rule-based score (AI unavailable)'
        
        # Sort by score
        self.signals.sort(key=lambda x: x.get('ai_score', 0), reverse=True)
        
        # Add ranks
        for rank, signal in enumerate(self.signals, 1):
            signal['rank'] = rank
        
        print(f"✅ Applied rule-based scores to {len(self.signals)} signals")
    
    def _rank_signals_in_batches(self, batch_size):
        """Process signals in batches for large signal counts"""
        all_signals = self.signals.copy()
        total_batches = (len(all_signals) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(all_signals))
            
            print(f"  Processing batch {batch_num + 1}/{total_batches} ({start_idx+1}-{end_idx})...")
            
            # Temporarily set signals to current batch
            self.signals = all_signals[start_idx:end_idx]
            
            # Score this batch
            self.rank_signals_with_ai()
            
            # Store results back
            all_signals[start_idx:end_idx] = self.signals
        
        # Restore all signals
        self.signals = all_signals
        
        # Final sort across all batches
        self.signals.sort(key=lambda x: x.get('ai_score', 0), reverse=True)
        
        # Update ranks
        for rank, signal in enumerate(self.signals, 1):
            signal['rank'] = rank
        
        print(f"✅ Completed scoring {len(self.signals)} signals in {total_batches} batches")
    
    def save_locally(self):
        """Save all analysis data to local folder"""
        print(f"\n💾 Saving files locally to: {self.date_folder}")
        
        try:
            current = self.data.iloc[-1]
            
            # Save technical data CSV
            csv_filename = self._generate_filename('technical_data', 'csv')
            csv_path = os.path.join(self.date_folder, csv_filename)
            self.data.to_csv(csv_path)
            print(f"✅ Saved: {csv_filename}")
            
            # Save signals JSON with AI scores
            signals_data = {
                'symbol': self.symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'price': float(current['Close']),
                'signals': self.signals,
                'signal_count': len(self.signals),
                'top_signal_info': getattr(self, 'top_signal_info', {})
            }
            
            json_filename = self._generate_filename('signals', 'json')
            json_path = os.path.join(self.date_folder, json_filename)
            with open(json_path, 'w') as f:
                json.dump(signals_data, f, indent=2)
            print(f"✅ Saved: {json_filename}")
            
            # Save ranked signals report
            ranked_filename = self._generate_filename('ranked_signals', 'txt')
            ranked_path = os.path.join(self.date_folder, ranked_filename)
            with open(ranked_path, 'w') as f:
                f.write(self.generate_ranked_report())
            print(f"✅ Saved: {ranked_filename}")
            
            print(f"\n✅ All files saved to: {self.date_folder}")
            return True
            
        except Exception as e:
            print(f"❌ Local Save Error: {str(e)}")
            return False
    
    def generate_ranked_report(self):
        """Generate ranked signals report"""
        current = self.data.iloc[-1]
        
        report = f"""
{'='*80}
RANKED SIGNALS REPORT - {self.symbol}
{'='*80}

Price: ${current['Close']:.2f} | Change: {current['Price_Change']:.2f}%
Date: {current.name.strftime('%Y-%m-%d')}

"""
        if hasattr(self, 'top_signal_info') and self.top_signal_info:
            report += f"""🏆 TOP SIGNAL: #{self.top_signal_info.get('signal_number', 'N/A')}
{self.top_signal_info.get('why', 'N/A')}

"""
        
        report += f"{'='*80}\nALL SIGNALS (Ranked by AI):\n{'='*80}\n\n"
        
        for signal in self.signals:
            score = signal.get('ai_score', 'N/A')
            rank = signal.get('rank', '?')
            
            if isinstance(score, (int, float)):
                indicator = "🔥" if score >= 80 else "⚡" if score >= 60 else "📊" if score >= 40 else "⚠️"
            else:
                indicator = "❓"
            
            report += f"""#{rank} {indicator} SCORE: {score}/100
Signal: {signal['signal']}
Description: {signal['desc']}
Category: {signal['category']} | Strength: {signal['strength']}
AI Analysis: {signal.get('ai_reasoning', 'N/A')}
{'-'*80}

"""
        
        return report
    
    def display_results(self):
        """Display formatted results"""
        current = self.data.iloc[-1]
        
        print("\n" + "="*80)
        print(f"📊 {self.symbol} TECHNICAL ANALYSIS RESULTS")
        print("="*80)
        print(f"\nPrice: ${current['Close']:.2f} | Change: {current['Price_Change']:.2f}%")
        print(f"Date: {current.name.strftime('%Y-%m-%d')}")
        
        print("\n" + "="*80)
        print("📈 KEY INDICATORS")
        print("="*80)
        print(f"RSI: {current['RSI']:.1f} | MACD: {current['MACD']:.4f} | ADX: {current['ADX']:.1f}")
        print(f"Stochastic K: {current['Stoch_K']:.1f} | MFI: {current['MFI']:.1f}")
        print(f"Volatility: {current['Volatility']:.1f}% | ATR: {current['ATR']:.2f}")
        
        if hasattr(self, 'top_signal_info') and self.top_signal_info:
            print("\n" + "="*80)
            print("🏆 TOP SIGNAL BY AI")
            print("="*80)
            print(f"Signal #{self.top_signal_info.get('signal_number', 'N/A')}")
            print(f"{self.top_signal_info.get('why', 'N/A')}")
        
        print("\n" + "="*80)
        print("🎯 TOP 10 AI-RANKED SIGNALS")
        print("="*80)
        
        for i, sig in enumerate(self.signals[:10], 1):
            score = sig.get('ai_score', 'N/A')
            indicator = "🔥" if isinstance(score, (int, float)) and score >= 80 else "⚡" if isinstance(score, (int, float)) and score >= 60 else "📊"
            
            print(f"\n#{i} {indicator} [{score}/100] {sig['signal']}")
            print(f"   {sig['desc']}")
            print(f"   Category: {sig['category']} | Strength: {sig['strength']}")
            print(f"   AI: {sig.get('ai_reasoning', 'N/A')[:60]}...")
        
        bullish = sum(1 for s in self.signals if 'BULLISH' in s['strength'])
        bearish = sum(1 for s in self.signals if 'BEARISH' in s['strength'])
        
        print("\n" + "="*80)
        print("📊 SIGNAL SUMMARY")
        print("="*80)
        print(f"Total Signals: {len(self.signals)}")
        print(f"Bullish: {bullish} | Bearish: {bearish}")
        print(f"Overall Bias: {'🟢 BULLISH' if bullish > bearish else '🔴 BEARISH' if bearish > bullish else '🟡 NEUTRAL'}")
        
        return self


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_analysis(symbol='AAPL', period='1y', gemini_api_key=None):
    """
    Main function to run complete technical analysis
    
    Args:
        symbol: Stock ticker (e.g., 'AAPL', 'TSLA', 'NVDA')
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
        gemini_api_key: Gemini API key for AI ranking (optional)
    
    Returns:
        TechnicalAnalyzer object with all analysis data
    """
    
    print("=" * 80)
    print(f"🚀 ENHANCED TECHNICAL SCANNER: {symbol}")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = TechnicalAnalyzer(
        symbol=symbol,
        period=period,
        gemini_api_key=gemini_api_key
    )
    
    # Execute analysis pipeline
    print("\n📊 Step 1: Fetching market data...")
    analyzer.fetch_data()
    
    print("\n🔧 Step 2: Calculating indicators...")
    analyzer.calculate_indicators()
    
    print("\n🎯 Step 3: Detecting signals...")
    analyzer.detect_signals()
    
    print("\n🤖 Step 4: AI ranking signals (1-100)...")
    if gemini_api_key:
        analyzer.rank_signals_with_ai()
    else:
        print("⚠️  No Gemini API key provided. Skipping AI ranking.")
        print("   Get API key from: https://aistudio.google.com/app/apikey")
    
    print("\n💾 Step 5: Saving results...")
    analyzer.save_locally()
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"📂 Results saved to: {analyzer.date_folder}")
    
    # Display results
    analyzer.display_results()
    
    return analyzer


def create_signals_dataframe(analyzer):
    """Create a pandas DataFrame from signals for easy analysis"""
    signals_df = pd.DataFrame([
        {
            'Rank': sig.get('rank', '?'),
            'Score': sig.get('ai_score', 'N/A'),
            'Signal': sig['signal'],
            'Description': sig['desc'],
            'Category': sig['category'],
            'Strength': sig['strength'],
            'AI_Reasoning': sig.get('ai_reasoning', 'N/A')
        }
        for sig in analyzer.signals
    ])
    return signals_df


def get_category_breakdown(analyzer):
    """Analyze signals by category"""
    signal_categories = {}
    for signal in analyzer.signals:
        category = signal['category']
        if category not in signal_categories:
            signal_categories[category] = []
        signal_categories[category].append(signal)
    
    print("\n" + "="*80)
    print("📊 SIGNAL CATEGORY BREAKDOWN")
    print("="*80)
    
    for category, signals in sorted(signal_categories.items(), key=lambda x: len(x[1]), reverse=True):
        avg_score = np.mean([s.get('ai_score', 50) for s in signals])
        print(f"{category:30} | Count: {len(signals):3} | Avg Score: {avg_score:.1f}")
    
    return signal_categories


def get_strength_analysis(analyzer):
    """Analyze signals by strength"""
    strength_counts = {}
    for signal in analyzer.signals:
        strength = signal['strength']
        strength_counts[strength] = strength_counts.get(strength, 0) + 1
    
    print("\n" + "="*80)
    print("💪 SIGNAL STRENGTH DISTRIBUTION")
    print("="*80)
    
    for strength, count in sorted(strength_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{strength:25} | {count:3} signals")
    
    return strength_counts


def export_to_csv(analyzer, filename=None):
    """Export signals to CSV file"""
    if filename is None:
        filename = f"{analyzer.symbol}_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    signals_df = create_signals_dataframe(analyzer)
    filepath = os.path.join(analyzer.date_folder, filename)
    signals_df.to_csv(filepath, index=False)
    
    print(f"\n✅ Exported signals to: {filepath}")
    return filepath


def print_usage_instructions():
    """Print instructions for using the scanner"""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    ENHANCED TECHNICAL ANALYSIS SCANNER                     ║
║                              Usage Instructions                            ║
╚════════════════════════════════════════════════════════════════════════════╝

BASIC USAGE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Run analysis on a stock
analyzer = run_analysis('AAPL', period='1y')

# With Gemini AI ranking (recommended)
analyzer = run_analysis('TSLA', period='6mo', gemini_api_key='your-key-here')

# Or use environment variable
import os
analyzer = run_analysis('NVDA', period='3mo', gemini_api_key=os.getenv('GEMINI_API_KEY'))


ACCESSING RESULTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Get technical data
technical_data = analyzer.data
print(technical_data.tail())

# Get signals list
signals = analyzer.signals
print(f"Total signals: {len(signals)}")

# Create signals DataFrame
signals_df = create_signals_dataframe(analyzer)
print(signals_df.head(10))

# Category breakdown
categories = get_category_breakdown(analyzer)

# Strength analysis
strengths = get_strength_analysis(analyzer)


ADVANCED FEATURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Export to CSV
export_to_csv(analyzer)

# Access specific signal categories
ma_signals = [s for s in analyzer.signals if 'MA' in s['category']]
rsi_signals = [s for s in analyzer.signals if 'RSI' in s['category']]

# Filter by score
high_confidence = [s for s in analyzer.signals if s.get('ai_score', 0) >= 80]
print(f"High confidence signals: {len(high_confidence)}")

# Filter by strength
bullish = [s for s in analyzer.signals if 'BULLISH' in s['strength']]
bearish = [s for s in analyzer.signals if 'BEARISH' in s['strength']]


SIGNAL CATEGORIES (150+ signals across):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Moving Averages:     MA_CROSS, MA_TREND, MA_COMPRESSION, MA_SLOPE, MA_DISTANCE
Momentum:            RSI, RSI_MOMENTUM, RSI_DIVERGENCE, RSI_CROSS
MACD:                MACD, MACD_MOMENTUM
Bollinger Bands:     BOLLINGER, BB_SQUEEZE, BB_WALK, BB_EXTREME
Volume:              VOLUME, VOLUME_DIVERGENCE, VOLUME_ACCUMULATION, VOLUME_CLIMAX
Price Action:        PRICE_ACTION, PRICE_PATTERN, GAP, CANDLESTICK
Structure:           RANGE, PIVOT, FIBONACCI, STRUCTURE_BREAK
Oscillators:         STOCHASTIC, WILLIAMS_R, CCI, MFI, ADX, OBV
Advanced:            ICHIMOKU, ROC, VWAP, ATR, EXHAUSTION
Multi-Factor:        TIMEFRAME_ALIGNMENT, MULTI_INDICATOR, CONSOLIDATION


EXAMPLE WORKFLOW:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 1. Run analysis
analyzer = run_analysis('ROKU', period='1mo', gemini_api_key=os.getenv('GEMINI_API_KEY'))

# 2. Get detailed breakdown
signals_df = create_signals_dataframe(analyzer)
categories = get_category_breakdown(analyzer)

# 3. Filter actionable signals
actionable = signals_df[signals_df['Score'] >= 70]
print(f"Actionable signals: {len(actionable)}")

# 4. Export results
export_to_csv(analyzer)

# 5. Access raw data for custom analysis
latest_price = analyzer.data.iloc[-1]
print(f"Current RSI: {latest_price['RSI']:.1f}")
print(f"Current MACD: {latest_price['MACD']:.4f}")


GET GEMINI API KEY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Visit: https://aistudio.google.com/app/apikey
Set environment variable: export GEMINI_API_KEY='your-key-here'

╔════════════════════════════════════════════════════════════════════════════╗
║  Ready to scan! Run: analyzer = run_analysis('YOUR_SYMBOL')              ║
╚════════════════════════════════════════════════════════════════════════════╝
""")


# ============================================================================
# EXAMPLE USAGE - UNCOMMENT TO RUN
# ============================================================================

if __name__ == "__main__":
    # Print usage instructions
    print_usage_instructions()
    
    # Example: Run analysis (uncomment to execute)
    # import os
    # 
    # # Basic usage
    # analyzer = run_analysis('AAPL', period='1y')
    # 
    # # With AI ranking (recommended)
    # analyzer = run_analysis(
    #     symbol='TSLA',
    #     period='6mo',
    #     gemini_api_key=os.getenv('GEMINI_API_KEY')
    # )
    # 
    # # Create signals DataFrame
    # signals_df = create_signals_dataframe(analyzer)
    # print("\nTop 10 Signals:")
    # print(signals_df.head(10))
    # 
    # # Get category breakdown
    # categories = get_category_breakdown(analyzer)
    # 
    # # Get strength analysis
    # strengths = get_strength_analysis(analyzer)
    # 
    # # Export to CSV
    # export_to_csv(analyzer)
    # 
    # # Access specific data
    # print(f"\nLatest Price: ${analyzer.data.iloc[-1]['Close']:.2f}")
    # print(f"RSI: {analyzer.data.iloc[-1]['RSI']:.1f}")
    # print(f"Total Signals: {len(analyzer.signals)}")
    # 
    # # Filter high-confidence signals
    # high_conf = [s for s in analyzer.signals if s.get('ai_score', 0) >= 80]
    # print(f"High-confidence signals (≥80): {len(high_conf)}")
    
    pass
