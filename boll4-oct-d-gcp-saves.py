# covert to 200 - loads
"""
Advanced Technical Analysis Scanner with GCP Storage & Gemini AI Analysis
150+ Alerts + Cloud Storage + Local Storage + AI-Powered Recommendations
""" 

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
import os
from google.cloud import storage
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
        
        # Initialize Gemini client if API key provided
        if self.gemini_api_key:
            self.genai_client = genai.Client(api_key=self.gemini_api_key)
        else:
            self.genai_client = None
        
        # Create local folder structure
        self._setup_local_folders()
    
    def _setup_local_folders(self):
        """Create local folder structure for saving analysis files"""
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        # Create main directory
        if not os.path.exists(self.local_save_dir):
            os.makedirs(self.local_save_dir)
            print(f"📁 Created main directory: {self.local_save_dir}")
        
        # Create date-specific subdirectory
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
    
    def save_locally(self):
        """Save all analysis data to local folder"""
        print(f"\n💾 Saving files locally to: {self.date_folder}")
        
        try:
            current = self.data.iloc[-1]
            
            # 1. Save full technical data CSV
            csv_filename = self._generate_filename('technical_data', 'csv')
            csv_path = os.path.join(self.date_folder, csv_filename)
            self.data.to_csv(csv_path)
            print(f"✅ Saved: {csv_filename}")
            
            # 2. Save signals JSON
            signals_data = {
                'symbol': self.symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'price': float(current['Close']),
                'change_pct': float(current['Price_Change']),
                'volume': int(current['Volume']),
                'indicators': {
                    'RSI': float(current['RSI']),
                    'MACD': float(current['MACD']),
                    'ADX': float(current['ADX']),
                    'Stochastic': float(current['Stoch_K']),
                    'CCI': float(current['CCI']),
                    'MFI': float(current['MFI']),
                    'BB_Position': float(current['BB_Position']),
                    'Volatility': float(current['Volatility'])
                },
                'moving_averages': {
                    'SMA_10': float(current['SMA_10']),
                    'SMA_20': float(current['SMA_20']),
                    'SMA_50': float(current['SMA_50']),
                    'SMA_200': float(current['SMA_200']) if not pd.isna(current['SMA_200']) else None,
                    'EMA_10': float(current['EMA_10']),
                    'EMA_20': float(current['EMA_20'])
                },
                'signals': self.signals,
                'signal_count': len(self.signals),
                'bullish_count': sum(1 for s in self.signals if 'BULLISH' in s['strength']),
                'bearish_count': sum(1 for s in self.signals if 'BEARISH' in s['strength'])
            }
            
            json_filename = self._generate_filename('signals', 'json')
            json_path = os.path.join(self.date_folder, json_filename)
            with open(json_path, 'w') as f:
                json.dump(signals_data, f, indent=2)
            print(f"✅ Saved: {json_filename}")
            
            # 3. Save summary report
            summary = self.generate_summary_text()
            txt_filename = self._generate_filename('summary', 'txt')
            txt_path = os.path.join(self.date_folder, txt_filename)
            with open(txt_path, 'w') as f:
                f.write(summary)
            print(f"✅ Saved: {txt_filename}")
            
            print(f"\n✅ All files saved to: {self.date_folder}")
            return True
            
        except Exception as e:
            print(f"❌ Local Save Error: {str(e)}")
            return False
        
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
        
        # Keltner Channels
        kc_ema = df['Close'].ewm(span=20, adjust=False).mean()
        kc_atr = df['ATR'].ewm(span=10, adjust=False).mean()
        df['KC_Middle'] = kc_ema
        df['KC_Upper'] = kc_ema + (kc_atr * 2)
        df['KC_Lower'] = kc_ema - (kc_atr * 2)

        # Donchian Channels
        df['DC_Upper'] = df['High'].rolling(window=20).max()
        df['DC_Lower'] = df['Low'].rolling(window=20).min()
        df['DC_Middle'] = (df['DC_Upper'] + df['DC_Lower']) / 2
        
        # Fibonacci Retracement Levels (using 100-day lookback)
        lookback_period = 100
        high_100 = df['High'].rolling(window=lookback_period).max()
        low_100 = df['Low'].rolling(window=lookback_period).min()
        
        price_range = high_100 - low_100
        df['Fib_0.236'] = high_100 - (price_range * 0.236)
        df['Fib_0.382'] = high_100 - (price_range * 0.382)
        df['Fib_0.500'] = high_100 - (price_range * 0.500)
        df['Fib_0.618'] = high_100 - (price_range * 0.618)
        df['Fib_0.786'] = high_100 - (price_range * 0.786)

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
        
        print("\n🎯 Scanning for 150+ Technical Alerts...")
        
        # === MOVING AVERAGE CROSSOVERS (10 alerts) ===
        
        if len(df) > 200 and prev['SMA_50'] <= prev['SMA_200'] and current['SMA_50'] > current['SMA_200']:
            signals.append({'signal': 'GOLDEN CROSS', 'desc': '50 MA crossed above 200 MA', 'strength': 'STRONG BULLISH', 'category': 'MA_CROSS'})
        
        if len(df) > 200 and prev['SMA_50'] >= prev['SMA_200'] and current['SMA_50'] < current['SMA_200']:
            signals.append({'signal': 'DEATH CROSS', 'desc': '50 MA crossed below 200 MA', 'strength': 'STRONG BEARISH', 'category': 'MA_CROSS'})
        
        if prev['Close'] <= prev['SMA_10'] and current['Close'] > current['SMA_10']:
            signals.append({'signal': 'PRICE ABOVE 10 MA', 'desc': 'Price crossed above 10-day MA', 'strength': 'BULLISH', 'category': 'MA_CROSS'})
        
        if prev['Close'] >= prev['SMA_10'] and current['Close'] < current['SMA_10']:
            signals.append({'signal': 'PRICE BELOW 10 MA', 'desc': 'Price crossed below 10-day MA', 'strength': 'BEARISH', 'category': 'MA_CROSS'})
        
        if abs(current['Close'] - current['SMA_10']) / current['SMA_10'] < 0.01:
            signals.append({'signal': 'PRICE AT 10 MA', 'desc': f"Price within 1% of 10 MA (${current['SMA_10']:.2f})", 'strength': 'WATCH', 'category': 'MA_PROXIMITY'})
        
        if prev['Close'] <= prev['SMA_20'] and current['Close'] > current['SMA_20']:
            signals.append({'signal': 'PRICE ABOVE 20 MA', 'desc': 'Price crossed above 20-day MA', 'strength': 'BULLISH', 'category': 'MA_CROSS'})
        
        if prev['Close'] >= prev['SMA_20'] and current['Close'] < current['SMA_20']:
            signals.append({'signal': 'PRICE BELOW 20 MA', 'desc': 'Price crossed below 20-day MA', 'strength': 'BEARISH', 'category': 'MA_CROSS'})
        
        if prev['EMA_10'] <= prev['EMA_20'] and current['EMA_10'] > current['EMA_20']:
            signals.append({'signal': '10/20 EMA BULL CROSS', 'desc': '10 EMA crossed above 20 EMA', 'strength': 'BULLISH', 'category': 'MA_CROSS'})
        
        if prev['EMA_10'] >= prev['EMA_20'] and current['EMA_10'] < current['EMA_20']:
            signals.append({'signal': '10/20 EMA BEAR CROSS', 'desc': '10 EMA crossed below 20 EMA', 'strength': 'BEARISH', 'category': 'MA_CROSS'})
        
        if prev['EMA_20'] <= prev['EMA_50'] and current['EMA_20'] > current['EMA_50']:
            signals.append({'signal': '20/50 EMA BULL CROSS', 'desc': '20 EMA crossed above 50 EMA', 'strength': 'BULLISH', 'category': 'MA_CROSS'})
        
        # === RSI ALERTS (10 alerts) ===
        
        if current['RSI'] < 30:
            signals.append({'signal': 'RSI OVERSOLD', 'desc': f"RSI at {current['RSI']:.1f}", 'strength': 'BULLISH', 'category': 'RSI'})
        
        if current['RSI'] > 70:
            signals.append({'signal': 'RSI OVERBOUGHT', 'desc': f"RSI at {current['RSI']:.1f}", 'strength': 'BEARISH', 'category': 'RSI'})
        
        if current['RSI'] < 20:
            signals.append({'signal': 'RSI EXTREME OVERSOLD', 'desc': f"RSI at {current['RSI']:.1f}", 'strength': 'STRONG BULLISH', 'category': 'RSI'})
        
        if current['RSI'] > 80:
            signals.append({'signal': 'RSI EXTREME OVERBOUGHT', 'desc': f"RSI at {current['RSI']:.1f}", 'strength': 'STRONG BEARISH', 'category': 'RSI'})
        
        if prev['RSI'] <= 50 and current['RSI'] > 50:
            signals.append({'signal': 'RSI ABOVE 50', 'desc': 'RSI crossed bullish threshold', 'strength': 'BULLISH', 'category': 'RSI'})
        
        if prev['RSI'] >= 50 and current['RSI'] < 50:
            signals.append({'signal': 'RSI BELOW 50', 'desc': 'RSI crossed bearish threshold', 'strength': 'BEARISH', 'category': 'RSI'})
        
        if len(df) > 20:
            if current['Close'] < df['Close'].iloc[-20] and current['RSI'] > df['RSI'].iloc[-20]:
                signals.append({'signal': 'RSI BULLISH DIVERGENCE', 'desc': 'Price down but RSI up', 'strength': 'BULLISH', 'category': 'DIVERGENCE'})
        
        if len(df) > 20:
            if current['Close'] > df['Close'].iloc[-20] and current['RSI'] < df['RSI'].iloc[-20]:
                signals.append({'signal': 'RSI BEARISH DIVERGENCE', 'desc': 'Price up but RSI down', 'strength': 'BEARISH', 'category': 'DIVERGENCE'})
        
        if prev['RSI'] >= 30 and current['RSI'] < 30:
            signals.append({'signal': 'RSI ENTERING OVERSOLD', 'desc': 'RSI just dropped below 30', 'strength': 'WATCH', 'category': 'RSI'})
        
        if prev['RSI'] >= 70 and current['RSI'] < 70:
            signals.append({'signal': 'RSI EXITING OVERBOUGHT', 'desc': 'RSI dropped from overbought', 'strength': 'BEARISH', 'category': 'RSI'})
        
        # === MACD ALERTS (10 alerts) ===
        
        if prev['MACD'] <= prev['MACD_Signal'] and current['MACD'] > current['MACD_Signal']:
            signals.append({'signal': 'MACD BULL CROSS', 'desc': 'MACD crossed above signal', 'strength': 'BULLISH', 'category': 'MACD'})
        
        if prev['MACD'] >= prev['MACD_Signal'] and current['MACD'] < current['MACD_Signal']:
            signals.append({'signal': 'MACD BEAR CROSS', 'desc': 'MACD crossed below signal', 'strength': 'BEARISH', 'category': 'MACD'})
        
        if prev['MACD'] <= 0 and current['MACD'] > 0:
            signals.append({'signal': 'MACD ABOVE ZERO', 'desc': 'MACD crossed into positive territory', 'strength': 'BULLISH', 'category': 'MACD'})
        
        if prev['MACD'] >= 0 and current['MACD'] < 0:
            signals.append({'signal': 'MACD BELOW ZERO', 'desc': 'MACD crossed into negative territory', 'strength': 'BEARISH', 'category': 'MACD'})
        
        if current['MACD_Hist'] > prev['MACD_Hist'] and prev['MACD_Hist'] > prev2['MACD_Hist']:
            signals.append({'signal': 'MACD MOMENTUM UP', 'desc': 'Histogram expanding bullish', 'strength': 'BULLISH', 'category': 'MACD'})
        
        if current['MACD_Hist'] < prev['MACD_Hist'] and prev['MACD_Hist'] < prev2['MACD_Hist']:
            signals.append({'signal': 'MACD MOMENTUM DOWN', 'desc': 'Histogram expanding bearish', 'strength': 'BEARISH', 'category': 'MACD'})
        
        if current['MACD_Hist'] > 0 and current['MACD_Hist'] > prev['MACD_Hist'] * 1.2:
            signals.append({'signal': 'MACD STRONG MOMENTUM', 'desc': 'Histogram accelerating up', 'strength': 'STRONG BULLISH', 'category': 'MACD'})
        
        if current['MACD_Hist'] < 0 and current['MACD_Hist'] < prev['MACD_Hist'] * 1.2:
            signals.append({'signal': 'MACD WEAK MOMENTUM', 'desc': 'Histogram accelerating down', 'strength': 'STRONG BEARISH', 'category': 'MACD'})
        
        if current['MACD_Signal'] > prev['MACD_Signal'] and prev['MACD_Signal'] > prev2['MACD_Signal']:
            signals.append({'signal': 'MACD SIGNAL RISING', 'desc': 'Signal line trending upward', 'strength': 'BULLISH', 'category': 'MACD'})
        
        if current['MACD'] > 0 and current['MACD_Signal'] > 0:
            signals.append({'signal': 'MACD FULLY BULLISH', 'desc': 'Both lines above zero', 'strength': 'BULLISH', 'category': 'MACD'})
        
        # === BOLLINGER BANDS (10 alerts) ===
        
        bb_width_avg = df['BB_Width'].tail(50).mean()
        if current['BB_Width'] < bb_width_avg * 0.7:
            signals.append({'signal': 'BB SQUEEZE', 'desc': 'Bands narrowing - breakout pending', 'strength': 'NEUTRAL', 'category': 'BOLLINGER'})
        
        if current['Close'] <= current['BB_Lower'] * 1.01:
            signals.append({'signal': 'AT LOWER BB', 'desc': f"Price at ${current['BB_Lower']:.2f}", 'strength': 'BULLISH', 'category': 'BOLLINGER'})
        
        if current['Close'] >= current['BB_Upper'] * 0.99:
            signals.append({'signal': 'AT UPPER BB', 'desc': f"Price at ${current['BB_Upper']:.2f}", 'strength': 'BEARISH', 'category': 'BOLLINGER'})
        
        if prev['Close'] <= prev['BB_Upper'] and current['Close'] > current['BB_Upper']:
            signals.append({'signal': 'BROKE UPPER BB', 'desc': 'Strong momentum or overextension', 'strength': 'BEARISH', 'category': 'BOLLINGER'})
        
        if prev['Close'] >= prev['BB_Lower'] and current['Close'] < current['BB_Lower']:
            signals.append({'signal': 'BROKE LOWER BB', 'desc': 'Oversold or strong selling', 'strength': 'BULLISH', 'category': 'BOLLINGER'})
        
        if current['BB_Width'] > prev['BB_Width'] * 1.1:
            signals.append({'signal': 'BB EXPANDING', 'desc': 'Volatility increasing', 'strength': 'VOLATILE', 'category': 'BOLLINGER'})
        
        if abs(current['Close'] - current['BB_Middle']) / current['BB_Middle'] < 0.005:
            signals.append({'signal': 'AT BB MIDDLE', 'desc': 'Price at midline', 'strength': 'NEUTRAL', 'category': 'BOLLINGER'})
        
        if current['BB_Position'] > 0.95:
            signals.append({'signal': 'BB TOP RANGE', 'desc': f"In top 5% of BB range", 'strength': 'BEARISH', 'category': 'BOLLINGER'})
        
        if current['BB_Position'] < 0.05:
            signals.append({'signal': 'BB BOTTOM RANGE', 'desc': f"In bottom 5% of BB range", 'strength': 'BULLISH', 'category': 'BOLLINGER'})
        
        if current['Close'] > current['BB_Upper'] * 0.98 and prev['Close'] > prev['BB_Upper'] * 0.98:
            signals.append({'signal': 'WALKING UPPER BAND', 'desc': 'Strong uptrend or overbought', 'strength': 'EXTREME', 'category': 'BOLLINGER'})
        
        # === VOLUME ALERTS (10 alerts) ===
        
        if current['Volume'] > current['Volume_MA_20'] * 2:
            signals.append({'signal': 'VOLUME SPIKE 2X', 'desc': f"Vol: {current['Volume']:,.0f}", 'strength': 'SIGNIFICANT', 'category': 'VOLUME'})
        
        if current['Volume'] > current['Volume_MA_20'] * 3:
            signals.append({'signal': 'EXTREME VOLUME 3X', 'desc': f"Vol: {current['Volume']:,.0f}", 'strength': 'VERY SIGNIFICANT', 'category': 'VOLUME'})
        
        if current['Volume'] < current['Volume_MA_20'] * 0.5:
            signals.append({'signal': 'LOW VOLUME', 'desc': 'Below average activity', 'strength': 'WEAK', 'category': 'VOLUME'})
        
        if current['Volume'] > prev['Volume'] and prev['Volume'] > prev2['Volume']:
            signals.append({'signal': 'VOLUME RISING', 'desc': 'Participation increasing', 'strength': 'WATCH', 'category': 'VOLUME'})
        
        if current['Price_Change'] > 2 and current['Volume'] > current['Volume_MA_20'] * 1.5:
            signals.append({'signal': 'VOLUME BREAKOUT', 'desc': 'High volume + price up', 'strength': 'STRONG BULLISH', 'category': 'VOLUME'})
        
        if current['Price_Change'] < -2 and current['Volume'] > current['Volume_MA_20'] * 1.5:
            signals.append({'signal': 'VOLUME SELLOFF', 'desc': 'High volume + price down', 'strength': 'STRONG BEARISH', 'category': 'VOLUME'})
        
        if current['OBV'] > prev['OBV'] and prev['OBV'] > prev2['OBV']:
            signals.append({'signal': 'OBV RISING', 'desc': 'Buying pressure increasing', 'strength': 'BULLISH', 'category': 'VOLUME'})
        
        if current['OBV'] < prev['OBV'] and prev['OBV'] < prev2['OBV']:
            signals.append({'signal': 'OBV FALLING', 'desc': 'Selling pressure increasing', 'strength': 'BEARISH', 'category': 'VOLUME'})
        
        if len(df) > 10 and current['Close'] < df['Close'].iloc[-10] and current['OBV'] > df['OBV'].iloc[-10]:
            signals.append({'signal': 'OBV BULL DIVERGENCE', 'desc': 'Price down, OBV up', 'strength': 'BULLISH', 'category': 'DIVERGENCE'})
        
        if current['Volume'] < current['Volume_MA_20'] * 0.3:
            signals.append({'signal': 'VOLUME DRYING UP', 'desc': 'Very low participation', 'strength': 'CAUTION', 'category': 'VOLUME'})
        
        # === STOCHASTIC & OTHER OSCILLATORS (10 alerts) ===
        
        if current['Stoch_K'] < 20:
            signals.append({'signal': 'STOCH OVERSOLD', 'desc': f"K: {current['Stoch_K']:.1f}", 'strength': 'BULLISH', 'category': 'STOCHASTIC'})
        
        if current['Stoch_K'] > 80:
            signals.append({'signal': 'STOCH OVERBOUGHT', 'desc': f"K: {current['Stoch_K']:.1f}", 'strength': 'BEARISH', 'category': 'STOCHASTIC'})
        
        if prev['Stoch_K'] <= prev['Stoch_D'] and current['Stoch_K'] > current['Stoch_D']:
            signals.append({'signal': 'STOCH BULL CROSS', 'desc': 'K crossed above D', 'strength': 'BULLISH', 'category': 'STOCHASTIC'})
        
        if prev['Stoch_K'] >= prev['Stoch_D'] and current['Stoch_K'] < current['Stoch_D']:
            signals.append({'signal': 'STOCH BEAR CROSS', 'desc': 'K crossed below D', 'strength': 'BEARISH', 'category': 'STOCHASTIC'})
        
        if current['Williams_R'] < -80:
            signals.append({'signal': 'WILLIAMS OVERSOLD', 'desc': f"W%R: {current['Williams_R']:.1f}", 'strength': 'BULLISH', 'category': 'WILLIAMS'})
        
        if current['Williams_R'] > -20:
            signals.append({'signal': 'WILLIAMS OVERBOUGHT', 'desc': f"W%R: {current['Williams_R']:.1f}", 'strength': 'BEARISH', 'category': 'WILLIAMS'})
        
        if current['CCI'] < -200:
            signals.append({'signal': 'CCI EXTREME OVERSOLD', 'desc': f"CCI: {current['CCI']:.1f}", 'strength': 'STRONG BULLISH', 'category': 'CCI'})
        
        if current['CCI'] > 200:
            signals.append({'signal': 'CCI EXTREME OVERBOUGHT', 'desc': f"CCI: {current['CCI']:.1f}", 'strength': 'STRONG BEARISH', 'category': 'CCI'})
        
        if prev['CCI'] <= 0 and current['CCI'] > 0:
            signals.append({'signal': 'CCI POSITIVE', 'desc': 'CCI crossed above zero', 'strength': 'BULLISH', 'category': 'CCI'})
        
        if prev['CCI'] >= 0 and current['CCI'] < 0:
            signals.append({'signal': 'CCI NEGATIVE', 'desc': 'CCI crossed below zero', 'strength': 'BEARISH', 'category': 'CCI'})
        
        # === MFI & MONEY FLOW (5 alerts) ===
        
        if current['MFI'] < 20:
            signals.append({'signal': 'MFI OVERSOLD', 'desc': f"MFI: {current['MFI']:.1f}", 'strength': 'BULLISH', 'category': 'MFI'})
        
        if current['MFI'] > 80:
            signals.append({'signal': 'MFI OVERBOUGHT', 'desc': f"MFI: {current['MFI']:.1f}", 'strength': 'BEARISH', 'category': 'MFI'})
        
        if prev['MFI'] <= 50 and current['MFI'] > 50:
            signals.append({'signal': 'MFI BULLISH', 'desc': 'Money flow crossed 50', 'strength': 'BULLISH', 'category': 'MFI'})
        
        if prev['MFI'] >= 50 and current['MFI'] < 50:
            signals.append({'signal': 'MFI BEARISH', 'desc': 'Money flow crossed below 50', 'strength': 'BEARISH', 'category': 'MFI'})
        
        if len(df) > 10 and current['Close'] < df['Close'].iloc[-10] and current['MFI'] > df['MFI'].iloc[-10]:
            signals.append({'signal': 'MFI BULL DIVERGENCE', 'desc': 'Price down, MFI up', 'strength': 'BULLISH', 'category': 'DIVERGENCE'})
        
        # === TREND STRENGTH (5 alerts) ===
        
        if current['ADX'] > 25:
            trend = 'UP' if current['Close'] > current['SMA_50'] else 'DOWN'
            signals.append({'signal': f"STRONG {trend}TREND", 'desc': f"ADX: {current['ADX']:.1f}", 'strength': 'TRENDING', 'category': 'TREND'})
        
        if current['ADX'] > 40:
            signals.append({'signal': 'VERY STRONG TREND', 'desc': f"ADX: {current['ADX']:.1f}", 'strength': 'EXTREME', 'category': 'TREND'})
        
        if current['ADX'] < 20:
            signals.append({'signal': 'WEAK TREND', 'desc': f"ADX: {current['ADX']:.1f} - ranging", 'strength': 'NEUTRAL', 'category': 'TREND'})
        
        if current['Plus_DI'] > current['Minus_DI'] and current['ADX'] > 25:
            signals.append({'signal': 'STRONG UPTREND CONFIRMED', 'desc': '+DI > -DI with high ADX', 'strength': 'BULLISH', 'category': 'TREND'})
        
        if current['Minus_DI'] > current['Plus_DI'] and current['ADX'] > 25:
            signals.append({'signal': 'STRONG DOWNTREND CONFIRMED', 'desc': '-DI > +DI with high ADX', 'strength': 'BEARISH', 'category': 'TREND'})
        
        # === PRICE ACTION (10 alerts) ===
        
        if current['Price_Change'] > 5:
            signals.append({'signal': 'LARGE GAIN', 'desc': f"+{current['Price_Change']:.1f}% today", 'strength': 'STRONG BULLISH', 'category': 'PRICE_ACTION'})
        
        if current['Price_Change'] < -5:
            signals.append({'signal': 'LARGE LOSS', 'desc': f"{current['Price_Change']:.1f}% today", 'strength': 'STRONG BEARISH', 'category': 'PRICE_ACTION'})
        
        if current['Price_Change'] > 10:
            signals.append({'signal': 'EXPLOSIVE MOVE UP', 'desc': f"+{current['Price_Change']:.1f}%", 'strength': 'EXTREME BULLISH', 'category': 'PRICE_ACTION'})
        
        if current['Price_Change'] < -10:
            signals.append({'signal': 'EXPLOSIVE MOVE DOWN', 'desc': f"{current['Price_Change']:.1f}%", 'strength': 'EXTREME BEARISH', 'category': 'PRICE_ACTION'})
        
        if current['Close'] > prev['High']:
            signals.append({'signal': 'HIGHER HIGH', 'desc': 'Breaking above yesterday', 'strength': 'BULLISH', 'category': 'PRICE_ACTION'})
        
        if current['Close'] < prev['Low']:
            signals.append({'signal': 'LOWER LOW', 'desc': 'Breaking below yesterday', 'strength': 'BEARISH', 'category': 'PRICE_ACTION'})
        
        daily_range = ((current['High'] - current['Low']) / current['Low']) * 100
        if daily_range > 5:
            signals.append({'signal': 'WIDE RANGE DAY', 'desc': f"Range: {daily_range:.1f}%", 'strength': 'VOLATILE', 'category': 'PRICE_ACTION'})
        
        if daily_range < 1:
            signals.append({'signal': 'NARROW RANGE DAY', 'desc': f"Range: {daily_range:.1f}%", 'strength': 'CONSOLIDATION', 'category': 'PRICE_ACTION'})
        
        body = abs(current['Close'] - current['Open'])
        full_range = current['High'] - current['Low']
        if full_range > 0 and body / full_range > 0.8:
            if current['Close'] > current['Open']:
                signals.append({'signal': 'STRONG BULLISH CANDLE', 'desc': 'Large body, small wicks', 'strength': 'BULLISH', 'category': 'PRICE_ACTION'})
            else:
                signals.append({'signal': 'STRONG BEARISH CANDLE', 'desc': 'Large body, small wicks', 'strength': 'BEARISH', 'category': 'PRICE_ACTION'})
        
        if current['Momentum'] > 0 and prev['Momentum'] > 0:
            signals.append({'signal': 'MOMENTUM BUILDING', 'desc': 'Consecutive positive momentum', 'strength': 'BULLISH', 'category': 'MOMENTUM'})
        
        # === 52-WEEK & RANGE ALERTS (10 alerts) ===
        
        if current['Close'] >= current['High_52w'] * 0.999:
            signals.append({'signal': '52-WEEK HIGH', 'desc': f"At ${current['Close']:.2f}", 'strength': 'STRONG BULLISH', 'category': 'RANGE'})
        
        if current['Close'] <= current['Low_52w'] * 1.001:
            signals.append({'signal': '52-WEEK LOW', 'desc': f"At ${current['Close']:.2f}", 'strength': 'STRONG BEARISH', 'category': 'RANGE'})
        
        distance_from_high = ((current['High_52w'] - current['Close']) / current['High_52w']) * 100
        if distance_from_high < 5:
            signals.append({'signal': 'NEAR 52W HIGH', 'desc': f"{distance_from_high:.1f}% below high", 'strength': 'BULLISH', 'category': 'RANGE'})
        
        distance_from_low = ((current['Close'] - current['Low_52w']) / current['Low_52w']) * 100
        if distance_from_low < 5:
            signals.append({'signal': 'NEAR 52W LOW', 'desc': f"{distance_from_low:.1f}% above low", 'strength': 'BEARISH', 'category': 'RANGE'})
        
        if current['Close'] >= current['High_20d'] * 0.999:
            signals.append({'signal': '20-DAY HIGH', 'desc': 'Breaking recent resistance', 'strength': 'BULLISH', 'category': 'RANGE'})
        
        if current['Close'] <= current['Low_20d'] * 1.001:
            signals.append({'signal': '20-DAY LOW', 'desc': 'Breaking recent support', 'strength': 'BEARISH', 'category': 'RANGE'})
        
        fifty_two_week_position = ((current['Close'] - current['Low_52w']) / (current['High_52w'] - current['Low_52w'])) * 100
        if fifty_two_week_position > 90:
            signals.append({'signal': 'TOP OF 52W RANGE', 'desc': f"At {fifty_two_week_position:.0f}% of range", 'strength': 'OVERBOUGHT', 'category': 'RANGE'})
        
        if fifty_two_week_position < 10:
            signals.append({'signal': 'BOTTOM OF 52W RANGE', 'desc': f"At {fifty_two_week_position:.0f}% of range", 'strength': 'OVERSOLD', 'category': 'RANGE'})
        
        if current['ROC_20'] > 20:
            signals.append({'signal': 'STRONG 20D MOMENTUM', 'desc': f"+{current['ROC_20']:.1f}% in 20 days", 'strength': 'STRONG BULLISH', 'category': 'MOMENTUM'})
        
        if current['ROC_20'] < -20:
            signals.append({'signal': 'WEAK 20D MOMENTUM', 'desc': f"{current['ROC_20']:.1f}% in 20 days", 'strength': 'STRONG BEARISH', 'category': 'MOMENTUM'})
        
        # === VOLATILITY ALERTS (5 alerts) ===
        
        if current['Volatility'] > 50:
            signals.append({'signal': 'HIGH VOLATILITY', 'desc': f"{current['Volatility']:.0f}% annualized", 'strength': 'CAUTION', 'category': 'VOLATILITY'})
        
        if current['Volatility'] > 80:
            signals.append({'signal': 'EXTREME VOLATILITY', 'desc': f"{current['Volatility']:.0f}% annualized", 'strength': 'HIGH RISK', 'category': 'VOLATILITY'})
        
        if current['Volatility'] < 20:
            signals.append({'signal': 'LOW VOLATILITY', 'desc': f"{current['Volatility']:.0f}% annualized", 'strength': 'CALM', 'category': 'VOLATILITY'})
        
        if current['ATR'] > df['ATR'].tail(50).mean() * 1.5:
            signals.append({'signal': 'ATR ELEVATED', 'desc': 'Above-average true range', 'strength': 'VOLATILE', 'category': 'VOLATILITY'})
        
        if current['ATR'] < df['ATR'].tail(50).mean() * 0.5:
            signals.append({'signal': 'ATR COMPRESSED', 'desc': 'Below-average true range', 'strength': 'QUIET', 'category': 'VOLATILITY'})
        
        # === ICHIMOKU ALERTS (5 alerts) ===
        
        if current['Close'] > current['Senkou_A'] and current['Close'] > current['Senkou_B']:
            signals.append({'signal': 'ABOVE CLOUD', 'desc': 'Ichimoku bullish', 'strength': 'BULLISH', 'category': 'ICHIMOKU'})
        
        if current['Close'] < current['Senkou_A'] and current['Close'] < current['Senkou_B']:
            signals.append({'signal': 'BELOW CLOUD', 'desc': 'Ichimoku bearish', 'strength': 'BEARISH', 'category': 'ICHIMOKU'})
        
        if prev['Tenkan'] <= prev['Kijun'] and current['Tenkan'] > current['Kijun']:
            signals.append({'signal': 'TENKAN/KIJUN CROSS', 'desc': 'Ichimoku bull signal', 'strength': 'BULLISH', 'category': 'ICHIMOKU'})
        
        if current['Senkou_A'] > current['Senkou_B']:
            signals.append({'signal': 'CLOUD BULLISH', 'desc': 'Senkou A above B', 'strength': 'BULLISH', 'category': 'ICHIMOKU'})
        
        if current['Senkou_A'] < current['Senkou_B']:
            signals.append({'signal': 'CLOUD BEARISH', 'desc': 'Senkou A below B', 'strength': 'BEARISH', 'category': 'ICHIMOKU'})
        
        # === MOVING AVERAGE TREND ALIGNMENT (5 alerts) ===
        
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
        
        if current['Dist_SMA_200'] > 20 and len(df) > 200:
            signals.append({'signal': 'EXTENDED FROM 200 SMA', 'desc': f"{current['Dist_SMA_200']:.1f}% above", 'strength': 'OVERBOUGHT', 'category': 'MA_TREND'})

        # === CANDLESTICK PATTERNS (10 alerts) ===
        body = abs(current['Close'] - current['Open'])
        full_range = current['High'] - current['Low']
        lower_wick = min(current['Open'], current['Close']) - current['Low']
        upper_wick = current['High'] - max(current['Open'], current['Close'])

        if full_range > 0 and lower_wick > body * 2 and upper_wick < body * 0.5:
            signals.append({'signal': 'HAMMER CANDLE', 'desc': 'Potential bottom reversal', 'strength': 'BULLISH', 'category': 'CANDLESTICK'})

        if full_range > 0 and upper_wick > body * 2 and lower_wick < body * 0.5:
            signals.append({'signal': 'INVERTED HAMMER', 'desc': 'Potential reversal, needs confirmation', 'strength': 'BULLISH', 'category': 'CANDLESTICK'})

        if current['Close'] > prev['High'] and current['Open'] < prev['Low'] and current['Close'] > current['Open']:
             signals.append({'signal': 'BULLISH ENGULFING', 'desc': 'Current candle engulfs previous', 'strength': 'STRONG BULLISH', 'category': 'CANDLESTICK'})

        if current['Close'] < prev['Low'] and current['Open'] > prev['High'] and current['Close'] < current['Open']:
            signals.append({'signal': 'BEARISH ENGULFING', 'desc': 'Current candle engulfs previous', 'strength': 'STRONG BEARISH', 'category': 'CANDLESTICK'})

        if len(df) > 3:
            c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
            if c1['Close'] < c1['Open'] and c2['Open'] < c1['Close'] and c3['Close'] > c2['Close'] and c3['Open'] > c2['Open'] and c3['Close'] > c1['Open']:
                signals.append({'signal': 'MORNING STAR', 'desc': 'Three-candle bullish reversal pattern', 'strength': 'STRONG BULLISH', 'category': 'CANDLESTICK'})
            if c1['Close'] > c1['Open'] and c2['Open'] > c1['Close'] and c3['Close'] < c2['Close'] and c3['Open'] < c2['Open'] and c3['Close'] < c1['Open']:
                signals.append({'signal': 'EVENING STAR', 'desc': 'Three-candle bearish reversal pattern', 'strength': 'STRONG BEARISH', 'category': 'CANDLESTICK'})

        if full_range > 0 and body / full_range < 0.1:
            signals.append({'signal': 'DOJI CANDLE', 'desc': 'Indecision in the market', 'strength': 'NEUTRAL', 'category': 'CANDLESTICK'})

        if full_range > 0 and body / full_range < 0.3 and upper_wick > body and lower_wick > body:
            signals.append({'signal': 'SPINNING TOP', 'desc': 'Potential for trend change', 'strength': 'NEUTRAL', 'category': 'CANDLESTICK'})

        if len(df) > 3:
            c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
            if (c1['Close'] > c1['Open'] and c2['Close'] > c2['Open'] and c3['Close'] > c3['Open'] and
                    c2['Open'] > c1['Open'] and c2['Close'] > c1['Close'] and
                    c3['Open'] > c2['Open'] and c3['Close'] > c2['Close']):
                signals.append({'signal': 'THREE WHITE SOLDIERS', 'desc': 'Strong bullish continuation', 'strength': 'STRONG BULLISH', 'category': 'CANDLESTICK'})
            if (c1['Close'] < c1['Open'] and c2['Close'] < c2['Open'] and c3['Close'] < c3['Open'] and
                    c2['Open'] < c1['Open'] and c2['Close'] < c1['Close'] and
                    c3['Open'] < c2['Open'] and c3['Close'] < c2['Close']):
                signals.append({'signal': 'THREE BLACK CROWS', 'desc': 'Strong bearish continuation', 'strength': 'STRONG BEARISH', 'category': 'CANDLESTICK'})

        # === GAP ANALYSIS (5 alerts) ===
        if current['Low'] > prev['High']:
            signals.append({'signal': 'GAP UP', 'desc': f"Gapped up from {prev['High']:.2f} to {current['Low']:.2f}", 'strength': 'BULLISH', 'category': 'GAP'})

        if current['High'] < prev['Low']:
            signals.append({'signal': 'GAP DOWN', 'desc': f"Gapped down from {prev['Low']:.2f} to {current['High']:.2f}", 'strength': 'BEARISH', 'category': 'GAP'})

        if 'GAP UP' in [s['signal'] for s in signals] and current['Close'] < current['Open']:
             signals.append({'signal': 'GAP UP FADING', 'desc': 'Gap up is being sold into', 'strength': 'BEARISH', 'category': 'GAP'})

        if 'GAP DOWN' in [s['signal'] for s in signals] and current['Close'] > current['Open']:
             signals.append({'signal': 'GAP DOWN REVERSING', 'desc': 'Gap down is being bought', 'strength': 'BULLISH', 'category': 'GAP'})

        # === VOLATILITY CHANNELS (10 alerts) ===
        if current['Close'] > current['KC_Upper']:
            signals.append({'signal': 'KC BREAKOUT UP', 'desc': 'Price broke above Keltner Channel', 'strength': 'BULLISH', 'category': 'VOL_CHANNEL'})
        if current['Close'] < current['KC_Lower']:
            signals.append({'signal': 'KC BREAKDOWN DOWN', 'desc': 'Price broke below Keltner Channel', 'strength': 'BEARISH', 'category': 'VOL_CHANNEL'})

        if current['Close'] > current['DC_Upper']:
            signals.append({'signal': 'DONCHIAN BREAKOUT UP', 'desc': 'Price broke above Donchian Channel', 'strength': 'STRONG BULLISH', 'category': 'VOL_CHANNEL'})
        if current['Close'] < current['DC_Lower']:
            signals.append({'signal': 'DONCHIAN BREAKDOWN DOWN', 'desc': 'Price broke below Donchian Channel', 'strength': 'STRONG BEARISH', 'category': 'VOL_CHANNEL'})

        if current['BB_Width'] < current['ATR'] / current['Close']:
             signals.append({'signal': 'TIGHT SQUEEZE (BBW < ATR%)', 'desc': 'Extreme low volatility, big move expected', 'strength': 'NEUTRAL', 'category': 'VOLATILITY'})

        # === INSIDE/OUTSIDE DAY (5 alerts) ===
        if current['High'] < prev['High'] and current['Low'] > prev['Low']:
            signals.append({'signal': 'INSIDE DAY (HARAMI)', 'desc': 'Volatility contraction, potential reversal/continuation', 'strength': 'NEUTRAL', 'category': 'PRICE_ACTION'})
        if current['High'] > prev['High'] and current['Low'] < prev['Low']:
            signals.append({'signal': 'OUTSIDE DAY (ENGULFING)', 'desc': 'Volatility expansion, potential reversal/continuation', 'strength': 'VOLATILE', 'category': 'PRICE_ACTION'})

        # === INDICATOR COMBINATIONS (10 alerts) ===
        if current['RSI'] > 50 and current['MACD'] > current['MACD_Signal'] and current['Close'] > current['SMA_20']:
            signals.append({'signal': 'BULLISH COMBO (RSI>50, MACD Cross, >SMA20)', 'desc': 'Multiple bullish signals confirming trend', 'strength': 'STRONG BULLISH', 'category': 'COMBO'})
        if current['RSI'] < 50 and current['MACD'] < current['MACD_Signal'] and current['Close'] < current['SMA_20']:
            signals.append({'signal': 'BEARISH COMBO (RSI<50, MACD Cross, <SMA20)', 'desc': 'Multiple bearish signals confirming trend', 'strength': 'STRONG BEARISH', 'category': 'COMBO'})

        if current['Stoch_K'] < 20 and current['RSI'] < 30:
            signals.append({'signal': 'DUAL OVERSOLD (Stoch & RSI)', 'desc': 'Both oscillators are in oversold territory', 'strength': 'STRONG BULLISH', 'category': 'COMBO'})
        if current['Stoch_K'] > 80 and current['RSI'] > 70:
            signals.append({'signal': 'DUAL OVERBOUGHT (Stoch & RSI)', 'desc': 'Both oscillators are in overbought territory', 'strength': 'STRONG BEARISH', 'category': 'COMBO'})

        if current['ADX'] > 25 and current['Close'] > current['SMA_50'] and current['Volume'] > current['Volume_MA_20']:
            signals.append({'signal': 'TRENDING UP WITH VOLUME', 'desc': 'ADX confirms trend, volume confirms strength', 'strength': 'STRONG BULLISH', 'category': 'COMBO'})
        if current['ADX'] > 25 and current['Close'] < current['SMA_50'] and current['Volume'] > current['Volume_MA_20']:
            signals.append({'signal': 'TRENDING DOWN WITH VOLUME', 'desc': 'ADX confirms trend, volume confirms strength', 'strength': 'STRONG BEARISH', 'category': 'COMBO'})

        # === ADDITIONAL MA SIGNALS (10) ===
        if current['EMA_5'] > current['EMA_10'] and current['EMA_10'] > current['EMA_20']:
            signals.append({'signal': 'SHORT-TERM EMA STACK BULLISH', 'desc': '5/10/20 EMAs aligned for uptrend', 'strength': 'BULLISH', 'category': 'MA_TREND'})
        if current['EMA_5'] < current['EMA_10'] and current['EMA_10'] < current['EMA_20']:
            signals.append({'signal': 'SHORT-TERM EMA STACK BEARISH', 'desc': '5/10/20 EMAs aligned for downtrend', 'strength': 'BEARISH', 'category': 'MA_TREND'})

        if prev['Close'] <= prev['VWAP'] and current['Close'] > current['VWAP']:
            signals.append({'signal': 'PRICE CROSSES VWAP UP', 'desc': 'Price crossed above Volume-Weighted Average Price', 'strength': 'BULLISH', 'category': 'VWAP'})
        if prev['Close'] >= prev['VWAP'] and current['Close'] < current['VWAP']:
            signals.append({'signal': 'PRICE CROSSES VWAP DOWN', 'desc': 'Price crossed below Volume-Weighted Average Price', 'strength': 'BEARISH', 'category': 'VWAP'})

        if current['Dist_SMA_50'] < -15:
            signals.append({'signal': 'OVERSOLD FROM 50 SMA', 'desc': "Price is >15% below 50-day SMA", 'strength': 'BULLISH', 'category': 'MA_PROXIMITY'})
        if current['Dist_SMA_50'] > 15:
            signals.append({'signal': 'OVERBOUGHT FROM 50 SMA', 'desc': "Price is >15% above 50-day SMA", 'strength': 'BEARISH', 'category': 'MA_PROXIMITY'})
        
        # === FIBONACCI RETRACEMENT (20 alerts) ===
        fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786]
        for level in fib_levels:
            fib_col = f'Fib_{level:.3f}'
            
            # Price crossing above Fibonacci level
            if prev['Close'] <= prev[fib_col] and current['Close'] > current[fib_col]:
                signals.append({'signal': f'CROSSING ABOVE FIB {level*100:.1f}%', 'desc': f'Price crossed above {level*100:.1f}% retracement level at ${current[fib_col]:.2f}', 'strength': 'BULLISH', 'category': 'FIBONACCI'})

            # Price crossing below Fibonacci level
            if prev['Close'] >= prev[fib_col] and current['Close'] < current[fib_col]:
                signals.append({'signal': f'CROSSING BELOW FIB {level*100:.1f}%', 'desc': f'Price crossed below {level*100:.1f}% retracement level at ${current[fib_col]:.2f}', 'strength': 'BEARISH', 'category': 'FIBONACCI'})

            # Price bouncing off Fibonacci level (as support)
            if prev['Low'] < prev[fib_col] and current['Low'] >= current[fib_col] and current['Close'] > current['Open']:
                 signals.append({'signal': f'BOUNCE FROM FIB {level*100:.1f}% (SUPPORT)', 'desc': f'Price bounced off {level*100:.1f}% support at ${current[fib_col]:.2f}', 'strength': 'STRONG BULLISH', 'category': 'FIBONACCI'})

            # Price bouncing off Fibonacci level (as resistance)
            if prev['High'] > prev[fib_col] and current['High'] <= current[fib_col] and current['Close'] < current['Open']:
                 signals.append({'signal': f'REJECTED AT FIB {level*100:.1f}% (RESISTANCE)', 'desc': f'Price rejected at {level*100:.1f}% resistance at ${current[fib_col]:.2f}', 'strength': 'STRONG BEARISH', 'category': 'FIBONACCI'})

        self.signals = signals
        
        print(f"✅ Detected {len(signals)} Active Signals")
        return signals

    
    def upload_to_gcp(self):
        """Upload data and alerts to GCP bucket using CLI authentication"""
        print(f"\n☁️  Uploading to GCP: {self.gcp_bucket}/daily...")
        
        try:
            # Initialize GCP Storage client - will use gcloud CLI credentials
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.gcp_bucket)
            
            # Prepare timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            date_folder = datetime.now().strftime('%Y-%m-%d')
            
            # 1. Upload full technical data CSV
            csv_data = self.data.to_csv()
            blob = bucket.blob(f'daily/{date_folder}/{self.symbol}_technical_data_{timestamp}.csv')
            blob.upload_from_string(csv_data, content_type='text/csv')
            print(f"✅ Uploaded: technical_data CSV")
            
            # 2. Upload signals JSON
            current = self.data.iloc[-1]
            signals_data = {
                'symbol': self.symbol,
                'timestamp': timestamp,
                'date': date_folder,
                'price': float(current['Close']),
                'change_pct': float(current['Price_Change']),
                'volume': int(current['Volume']),
                'indicators': {
                    'RSI': float(current['RSI']),
                    'MACD': float(current['MACD']),
                    'ADX': float(current['ADX']),
                    'Stochastic': float(current['Stoch_K']),
                    'CCI': float(current['CCI']),
                    'MFI': float(current['MFI']),
                    'BB_Position': float(current['BB_Position']),
                    'Volatility': float(current['Volatility'])
                },
                'moving_averages': {
                    'SMA_10': float(current['SMA_10']),
                    'SMA_20': float(current['SMA_20']),
                    'SMA_50': float(current['SMA_50']),
                    'SMA_200': float(current['SMA_200']) if not pd.isna(current['SMA_200']) else None,
                    'EMA_10': float(current['EMA_10']),
                    'EMA_20': float(current['EMA_20'])
                },
                'signals': self.signals,
                'signal_count': len(self.signals),
                'bullish_count': sum(1 for s in self.signals if 'BULLISH' in s['strength']),
                'bearish_count': sum(1 for s in self.signals if 'BEARISH' in s['strength'])
            }
            
            blob = bucket.blob(f'daily/{date_folder}/{self.symbol}_signals_{timestamp}.json')
            blob.upload_from_string(json.dumps(signals_data, indent=2), content_type='application/json')
            print(f"✅ Uploaded: signals JSON ({len(self.signals)} signals)")
            
            # 3. Upload summary report
            summary = self.generate_summary_text()
            blob = bucket.blob(f'daily/{date_folder}/{self.symbol}_summary_{timestamp}.txt')
            blob.upload_from_string(summary, content_type='text/plain')
            print(f"✅ Uploaded: summary report")
            
            print(f"\n✅ All files uploaded to gs://{self.gcp_bucket}/daily/{date_folder}/")
            return True
            
        except Exception as e:
            print(f"❌ GCP Upload Error: {str(e)}")
            return False
    
    def analyze_with_gemini(self):
        """Use Gemini AI to analyze signals and provide recommendations"""
        if not self.genai_client:
            print("\n⚠️  Gemini API key not provided. Skipping AI analysis.")
            return None
        
        print("\n🤖 Analyzing signals with Gemini AI...")
        
        try:
            current = self.data.iloc[-1]
            
            # Prepare comprehensive prompt for Gemini
            prompt = f"""
You are an expert technical analyst. Analyze the following technical signals for {self.symbol} and provide actionable insights.

CURRENT PRICE: ${current['Close']:.2f}
DAILY CHANGE: {current['Price_Change']:.2f}%
DATE: {current.name.strftime('%Y-%m-%d')}

KEY INDICATORS:
- RSI: {current['RSI']:.1f}
- MACD: {current['MACD']:.4f}
- Stochastic: {current['Stoch_K']:.1f}
- ADX: {current['ADX']:.1f}
- CCI: {current['CCI']:.1f}
- MFI: {current['MFI']:.1f}
- Volatility: {current['Volatility']:.1f}%

MOVING AVERAGES:
- Price vs 10 SMA: {current['Dist_SMA_10']:.1f}%
- Price vs 20 SMA: {current['Dist_SMA_20']:.1f}%
- Price vs 50 SMA: {current['Dist_SMA_50']:.1f}%

ACTIVE SIGNALS ({len(self.signals)} total):
"""
            # Add all signals to prompt
            for i, sig in enumerate(self.signals, 1):
                prompt += f"\n{i}. [{sig['category']}] {sig['signal']} - {sig['desc']} ({sig['strength']})"
            
            prompt += """

Please provide:
1. STRONGEST SIGNAL: Identify the single most actionable signal and explain why
2. OVERALL BIAS: Bullish, Bearish, or Neutral with confidence level
3. KEY LEVELS: Important support/resistance levels to watch
4. RISK ASSESSMENT: What are the main risks right now?
5. TRADING RECOMMENDATION: Should one buy, sell, hold, or wait? Include entry/exit strategy
6. TIMEFRAME: Best timeframe for this analysis (short-term/medium-term/long-term)

Be specific, concise, and actionable. Focus on the most important signals."""

            # Call Gemini API using new syntax
            response = self.genai_client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt
            )
            
            print("\n" + "="*80)
            print("🤖 GEMINI AI ANALYSIS")
            print("="*80)
            print(response.text)
            print("="*80)
            
            return response.text
            
        except Exception as e:
            print(f"❌ Gemini API Error: {str(e)}")
            return None
    
    def upload_gemini_analysis_to_gcp(self, analysis):
        """Upload Gemini analysis to GCP"""
        if not analysis:
            return
        
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.gcp_bucket)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            date_folder = datetime.now().strftime('%Y-%m-%d')
            
            # Create analysis document
            analysis_data = {
                'symbol': self.symbol,
                'timestamp': timestamp,
                'date': date_folder,
                'analysis': analysis,
                'signal_count': len(self.signals),
                'signals_analyzed': self.signals
            }
            
            # Upload as JSON
            blob = bucket.blob(f'daily/{date_folder}/{self.symbol}_gemini_analysis_{timestamp}.json')
            blob.upload_from_string(json.dumps(analysis_data, indent=2), content_type='application/json')
            
            # Also upload as readable text
            blob_txt = bucket.blob(f'daily/{date_folder}/{self.symbol}_gemini_analysis_{timestamp}.txt')
            blob_txt.upload_from_string(analysis, content_type='text/plain')
            
            print(f"✅ Gemini analysis uploaded to GCP")
            
        except Exception as e:
            print(f"❌ Error uploading Gemini analysis: {str(e)}")
    
    def save_gemini_analysis_locally(self, analysis):
        """Save Gemini analysis to local folder"""
        if not analysis:
            return
        
        try:
            # Create analysis document
            analysis_data = {
                'symbol': self.symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'analysis': analysis,
                'signal_count': len(self.signals),
                'signals_analyzed': self.signals
            }
            
            # Save as JSON
            json_filename = self._generate_filename('gemini_analysis', 'json')
            json_path = os.path.join(self.date_folder, json_filename)
            with open(json_path, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            print(f"✅ Saved: {json_filename}")
            
            # Also save as readable text
            txt_filename = self._generate_filename('gemini_analysis', 'txt')
            txt_path = os.path.join(self.date_folder, txt_filename)
            with open(txt_path, 'w') as f:
                f.write(f"GEMINI AI ANALYSIS - {self.symbol}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                f.write(analysis)
            print(f"✅ Saved: {txt_filename}")
            
        except Exception as e:
            print(f"❌ Error saving Gemini analysis locally: {str(e)}")
    
    def generate_summary_text(self):
        """Generate text summary for upload"""
        current = self.data.iloc[-1]
        
        summary = f"""
{'='*80}
TECHNICAL ANALYSIS SUMMARY - {self.symbol}
{'='*80}

DATE: {current.name.strftime('%Y-%m-%d')}
PRICE: ${current['Close']:.2f}
CHANGE: {current['Price_Change']:.2f}%
VOLUME: {current['Volume']:,.0f}

MOVING AVERAGES:
  10 SMA:  ${current['SMA_10']:.2f} ({current['Dist_SMA_10']:.1f}%)
  20 SMA:  ${current['SMA_20']:.2f} ({current['Dist_SMA_20']:.1f}%)
  50 SMA:  ${current['SMA_50']:.2f} ({current['Dist_SMA_50']:.1f}%)
  200 SMA: ${current['SMA_200']:.2f} ({current['Dist_SMA_200']:.1f}%)

KEY OSCILLATORS:
  RSI:        {current['RSI']:.1f}
  Stochastic: {current['Stoch_K']:.1f}
  CCI:        {current['CCI']:.1f}
  Williams:   {current['Williams_R']:.1f}
  MFI:        {current['MFI']:.1f}

MOMENTUM & TREND:
  MACD:       {current['MACD']:.4f}
  ADX:        {current['ADX']:.1f}
  Volatility: {current['Volatility']:.1f}%

BOLLINGER BANDS:
  Upper:  ${current['BB_Upper']:.2f}
  Middle: ${current['BB_Middle']:.2f}
  Lower:  ${current['BB_Lower']:.2f}

ACTIVE SIGNALS ({len(self.signals)} total):
"""
        
        for i, sig in enumerate(self.signals, 1):
            summary += f"\n{i}. {sig['signal']}\n   {sig['desc']} - {sig['strength']}\n"
        
        bullish = sum(1 for s in self.signals if 'BULLISH' in s['strength'])
        bearish = sum(1 for s in self.signals if 'BEARISH' in s['strength'])
        
        summary += f"""
SIGNAL BREAKDOWN:
  Bullish: {bullish}
  Bearish: {bearish}
  Neutral: {len(self.signals) - bullish - bearish}

OVERALL BIAS: {'BULLISH' if bullish > bearish else 'BEARISH' if bearish > bullish else 'NEUTRAL'}

{'='*80}
"""
        return summary
    
    def print_summary(self):
        """Print summary to console"""
        current = self.data.iloc[-1]
        
        print("\n" + "=" * 80)
        print(f"📊 TECHNICAL ANALYSIS SUMMARY - {self.symbol}")
        print("=" * 80)
        
        print(f"\n💰 Current Price: ${current['Close']:.2f}")
        print(f"📅 Date: {current.name.strftime('%Y-%m-%d')}")
        print(f"📈 Change: {current['Price_Change']:.2f}%")
        
        print("\n📈 Moving Averages:")
        print(f"   10 SMA: ${current['SMA_10']:.2f} ({current['Dist_SMA_10']:.1f}%)")
        print(f"   20 SMA: ${current['SMA_20']:.2f} ({current['Dist_SMA_20']:.1f}%)")
        print(f"   50 SMA: ${current['SMA_50']:.2f} ({current['Dist_SMA_50']:.1f}%)")
        
        print("\n📊 Key Indicators:")
        print(f"   RSI: {current['RSI']:.1f}")
        print(f"   MACD: {current['MACD']:.4f}")
        print(f"   ADX: {current['ADX']:.1f}")
        
        bullish = sum(1 for s in self.signals if 'BULLISH' in s['strength'])
        bearish = sum(1 for s in self.signals if 'BEARISH' in s['strength'])
        
        print(f"\n🎯 Signal Summary:")
        print(f"   Total Signals: {len(self.signals)}")
        print(f"   Bullish: {bullish}")
        print(f"   Bearish: {bearish}")
        
        if bullish > bearish * 1.5:
            bias = "🟢 STRONG BULLISH"
        elif bullish > bearish:
            bias = "🟢 BULLISH"
        elif bearish > bullish * 1.5:
            bias = "🔴 STRONG BEARISH"
        elif bearish > bullish:
            bias = "🔴 BEARISH"
        else:
            bias = "🟡 NEUTRAL"
        
        print(f"   Overall Bias: {bias}")
        print("\n" + "=" * 80)

def main():
    """Main execution with GCP and Gemini integration"""
    
    # Configuration
    SYMBOL = 'DIA'
    PERIOD = '1y'
    GCP_BUCKET = 'ttb-bucket1'
    
    # Set your Gemini API key here or via environment variable
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Get from environment
    # Or set directly: GEMINI_API_KEY = 'your-api-key-here'
    
    print("=" * 80)
    print("🚀 TECHNICAL SCANNER with GCP & GEMINI AI")
    print("=" * 80)
    
    try:
        # Initialize analyzer
        analyzer = TechnicalAnalyzer(
            symbol=SYMBOL, 
            period=PERIOD,
            gcp_bucket=GCP_BUCKET,
            gemini_api_key=GEMINI_API_KEY
        )
        
        # Step 1: Fetch data
        analyzer.fetch_data()
        
        # Step 2: Calculate indicators
        analyzer.calculate_indicators()
        
        # Step 3: Detect signals
        analyzer.detect_signals()
        
        # Step 4: Print summary
        analyzer.print_summary()
        
        # Step 5: Save files locally
        save_success = analyzer.save_locally()
        
        # Step 6: Upload to GCP (optional - can be disabled)
        try:
            upload_success = analyzer.upload_to_gcp()
        except Exception as e:
            print(f"\n⚠️  GCP upload skipped or failed: {str(e)}")
        
        # Step 7: Get Gemini AI analysis
        if GEMINI_API_KEY:
            gemini_analysis = analyzer.analyze_with_gemini()
            
            # Step 8: Save Gemini analysis locally
            if gemini_analysis:
                analyzer.save_gemini_analysis_locally(gemini_analysis)
                
                # Step 9: Upload Gemini analysis to GCP (optional)
                try:
                    analyzer.upload_gemini_analysis_to_gcp(gemini_analysis)
                except Exception as e:
                    print(f"⚠️  GCP upload of Gemini analysis skipped: {str(e)}")
        else:
            print("\n⚠️  Set GEMINI_API_KEY environment variable for AI analysis")
        
        print("\n✅ Analysis Complete!")
        print(f"📂 Files saved in: {analyzer.date_folder}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


"""
SETUP INSTRUCTIONS:
====================

1. Install required packages:
   pip install yfinance pandas numpy google-cloud-storage google-genai

2. Authenticate with GCP CLI (you already have this):
   gcloud auth application-default login
   
   This will set up Application Default Credentials automatically.
   
   Alternatively, set your project:
   gcloud config set project YOUR_PROJECT_ID
   
3. Set Gemini API key:
   - Get API key from https://aistudio.google.com/app/apikey
   - Set environment variable:
     export GEMINI_API_KEY="your-api-key-here"
   
   Or on Windows:
     set GEMINI_API_KEY=your-api-key-here
   
4. Ensure GCP bucket exists:
   - Bucket name: ttb-bucket1
   - Folder will be created automatically: daily/
   
   Check if bucket exists:
   gsutil ls gs://ttb-bucket1
   
5. Run the script:
   python scanner.py

OUTPUTS:
========

LOCAL STORAGE (NEW!):
- Folder: ./technical_analysis_data/YYYY-MM-DD/
- Files created each run:
  * YYYY-MM-DD-SYMBOL-technical_data-HHMMSS.csv
  * YYYY-MM-DD-SYMBOL-signals-HHMMSS.json
  * YYYY-MM-DD-SYMBOL-summary-HHMMSS.txt
  * YYYY-MM-DD-SYMBOL-gemini_analysis-HHMMSS.json
  * YYYY-MM-DD-SYMBOL-gemini_analysis-HHMMSS.txt

Example filenames:
  2025-10-21-ORCL-technical_data-143052.csv
  2025-10-21-ORCL-signals-143052.json
  2025-10-21-ORCL-summary-143052.txt
  2025-10-21-ORCL-gemini_analysis-143052.json
  2025-10-21-ORCL-gemini_analysis-143052.txt

GCP Storage (optional): gs://ttb-bucket1/daily/YYYY-MM-DD/
  - {SYMBOL}_technical_data_{timestamp}.csv  (full indicator data)
  - {SYMBOL}_signals_{timestamp}.json         (all detected signals)
  - {SYMBOL}_summary_{timestamp}.txt          (readable summary)
  - {SYMBOL}_gemini_analysis_{timestamp}.json (AI analysis JSON)
  - {SYMBOL}_gemini_analysis_{timestamp}.txt  (AI analysis readable)

GEMINI AI PROVIDES:
===================
1. Strongest signal identification
2. Overall market bias (bullish/bearish/neutral)
3. Key support/resistance levels
4. Risk assessment
5. Trading recommendations with entry/exit strategy
6. Optimal timeframe for trading

FOLDER STRUCTURE:
=================
technical_analysis_data/
├── 2025-10-21/
│   ├── 2025-10-21-ORCL-technical_data-143052.csv
│   ├── 2025-10-21-ORCL-signals-143052.json
│   ├── 2025-10-21-ORCL-summary-143052.txt
│   ├── 2025-10-21-ORCL-gemini_analysis-143052.json
│   └── 2025-10-21-ORCL-gemini_analysis-143052.txt
├── 2025-10-22/
│   ├── 2025-10-22-AAPL-technical_data-091523.csv
│   └── ...
└── 2025-10-23/
    └── ...

FEATURES:
=========
✅ 150+ Technical signals including:
   - Moving Average Crossovers (Golden/Death Cross)
   - RSI Oversold/Overbought + Divergences
   - MACD Bull/Bear Crosses
   - Bollinger Band Squeezes & Breakouts
   - Volume Spikes & OBV Divergences
   - Stochastic, Williams %R, CCI signals
   - ADX Trend Strength
   - MFI Money Flow analysis
   - 52-week Highs/Lows
   - Volatility & ATR alerts
   - Ichimoku Cloud signals
   - Candlestick Patterns (Hammer, Engulfing, Doji, etc.)
   - Gap Analysis
   - Keltner & Donchian Channel Breakouts
   - Price Action patterns
   - And much more!

✅ Local file storage with organized date folders
✅ Standardized filename format: YYYY-MM-DD-SYMBOL-type-HHMMSS.ext
✅ Optional GCP cloud storage backup
✅ Gemini AI analysis with actionable recommendations
✅ Complete technical indicator calculations
✅ JSON & TXT output formats

TROUBLESHOOTING:
================
GCP Authentication Error:
  Run: gcloud auth application-default login
  Or: gcloud auth login

Gemini API Error:
  - Verify API key is correct
  - Check you have API access enabled
  - Using latest model: 'gemini-2.0-flash-exp'
  - New SDK: from google import genai

Bucket Permission Error:
  - Ensure you have write permissions to ttb-bucket1
  - Check with: gsutil ls -L gs://ttb-bucket1

Package Installation:
  pip install google-genai  (NEW SDK - not google-generativeai)

Local Folder Permissions:
  - Script creates ./technical_analysis_data/ automatically
  - Ensure you have write permissions in current directory

CUSTOMIZATION:
==============
Change the stock symbol in main():
  SYMBOL = 'ROKU'  # Change to any ticker
  PERIOD = '1y'    # '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'

Change the local save directory:
  analyzer = TechnicalAnalyzer(
      symbol=SYMBOL,
      local_save_dir='my_custom_folder'  # Default: 'technical_analysis_data'
  )
"""