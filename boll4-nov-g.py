"""
Advanced Technical Analysis Scanner - 200+ Signals
Complete trading analysis with AI recommendations and options strategies
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

class AdvancedTechnicalAnalyzer:
    def __init__(self, symbol, period='1y', gcp_bucket='ttb-bucket1', 
                 gemini_api_key=None, local_save_dir='technical_analysis_data'):
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
        """Create local folder structure"""
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        if not os.path.exists(self.local_save_dir):
            os.makedirs(self.local_save_dir)
            print(f"📁 Created main directory: {self.local_save_dir}")
        
        self.date_folder = os.path.join(self.local_save_dir, date_str)
        if not os.path.exists(self.date_folder):
            os.makedirs(self.date_folder)
            print(f"📁 Created date folder: {self.date_folder}")
    
    def _generate_filename(self, file_type, extension):
        """Generate standardized filename"""
        date_str = datetime.now().strftime('%Y-%m-%d')
        timestamp = datetime.now().strftime('%H%M%S')
        return f"{date_str}-{self.symbol}-{file_type}-{timestamp}.{extension}"
    
    def fetch_data(self):
        """Fetch stock data"""
        print(f"📊 Fetching data for {self.symbol}...")
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(period=self.period)
        
        # Get options data for strategy recommendations
        try:
            self.options_dates = ticker.options
            self.ticker_info = ticker.info
        except:
            self.options_dates = []
            self.ticker_info = {}
        
        if self.data.empty:
            raise ValueError(f"No data found for {self.symbol}")
        
        print(f"✅ Fetched {len(self.data)} days of data")
        return self.data
    
    def calculate_indicators(self):
        """Calculate comprehensive technical indicators"""
        df = self.data.copy()
        
        print("\n🔧 Calculating 50+ Technical Indicators...")
        
        # Moving Averages (Extended)
        for period in [3, 5, 8, 10, 13, 20, 21, 30, 50, 100, 150, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # RSI (Multiple periods)
        for period in [9, 14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        df['RSI'] = df['RSI_14']  # Default RSI
        
        # MACD (Multiple configurations)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Fast MACD
        exp1_fast = df['Close'].ewm(span=5, adjust=False).mean()
        exp2_fast = df['Close'].ewm(span=35, adjust=False).mean()
        df['MACD_Fast'] = exp1_fast - exp2_fast
        df['MACD_Fast_Signal'] = df['MACD_Fast'].ewm(span=5, adjust=False).mean()
        
        # Bollinger Bands (Multiple periods)
        for period in [10, 20, 30]:
            bb_middle = df['Close'].rolling(window=period).mean()
            bb_std = df['Close'].rolling(window=period).std()
            df[f'BB_{period}_Middle'] = bb_middle
            df[f'BB_{period}_Upper'] = bb_middle + (bb_std * 2)
            df[f'BB_{period}_Lower'] = bb_middle - (bb_std * 2)
            df[f'BB_{period}_Width'] = df[f'BB_{period}_Upper'] - df[f'BB_{period}_Lower']
            df[f'BB_{period}_Position'] = (df['Close'] - df[f'BB_{period}_Lower']) / (df[f'BB_{period}_Upper'] - df[f'BB_{period}_Lower'])
        
        # Default BB
        df['BB_Middle'] = df['BB_20_Middle']
        df['BB_Upper'] = df['BB_20_Upper']
        df['BB_Lower'] = df['BB_20_Lower']
        df['BB_Width'] = df['BB_20_Width']
        df['BB_Position'] = df['BB_20_Position']
        
        # Keltner Channels
        df['KC_Middle'] = df['EMA_20']
        df['ATR_20'] = self._calculate_atr(df, 20)
        df['KC_Upper'] = df['KC_Middle'] + (df['ATR_20'] * 2)
        df['KC_Lower'] = df['KC_Middle'] - (df['ATR_20'] * 2)
        
        # Donchian Channels
        df['DC_Upper'] = df['High'].rolling(window=20).max()
        df['DC_Lower'] = df['Low'].rolling(window=20).min()
        df['DC_Middle'] = (df['DC_Upper'] + df['DC_Lower']) / 2
        
        # Stochastic (Multiple periods)
        for period in [5, 14, 21]:
            low_n = df['Low'].rolling(window=period).min()
            high_n = df['High'].rolling(window=period).max()
            df[f'Stoch_{period}_K'] = 100 * ((df['Close'] - low_n) / (high_n - low_n))
            df[f'Stoch_{period}_D'] = df[f'Stoch_{period}_K'].rolling(window=3).mean()
        
        df['Stoch_K'] = df['Stoch_14_K']
        df['Stoch_D'] = df['Stoch_14_D']
        
        # ATR (Multiple periods)
        for period in [7, 14, 21]:
            df[f'ATR_{period}'] = self._calculate_atr(df, period)
        
        df['ATR'] = df['ATR_14']
        
        # ADX and DI
        df = self._calculate_adx(df)
        
        # CCI
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        
        # Williams %R (Multiple periods)
        for period in [14, 21]:
            high_n = df['High'].rolling(window=period).max()
            low_n = df['Low'].rolling(window=period).min()
            df[f'Williams_R_{period}'] = -100 * ((high_n - df['Close']) / (high_n - low_n))
        
        df['Williams_R'] = df['Williams_R_14']
        
        # OBV and Volume indicators
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_MA_50'] = df['Volume'].rolling(window=50).mean()
        
        # Force Index
        df['Force_Index'] = df['Close'].diff() * df['Volume']
        df['Force_Index_13'] = df['Force_Index'].ewm(span=13, adjust=False).mean()
        
        # Ease of Movement
        distance = ((df['High'] + df['Low']) / 2 - (df['High'].shift(1) + df['Low'].shift(1)) / 2)
        box_ratio = df['Volume'] / (df['High'] - df['Low'])
        df['EMV'] = distance / box_ratio
        df['EMV_14'] = df['EMV'].rolling(window=14).mean()
        
        # VWAP
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        # ROC (Multiple periods)
        for period in [5, 10, 20, 30]:
            df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
        
        # MFI
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        mfi_ratio = positive_flow / negative_flow
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        
        # Ichimoku Cloud
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
        df['Chikou'] = df['Close'].shift(-26)
        
        # Parabolic SAR
        df = self._calculate_sar(df)
        
        # Momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        df['Momentum_20'] = df['Close'] - df['Close'].shift(20)
        
        # Volatility
        df['Volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
        df['Volatility_10'] = df['Close'].pct_change().rolling(10).std() * np.sqrt(252) * 100
        
        # Historical Volatility Ratio
        df['HV_Ratio'] = df['Volatility_10'] / df['Volatility']
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change() * 100
        df['Price_Change_3d'] = ((df['Close'] - df['Close'].shift(3)) / df['Close'].shift(3)) * 100
        df['Price_Change_5d'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
        df['Price_Change_10d'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # Highs and Lows (Multiple periods)
        for period in [5, 10, 20, 50, 100, 252]:
            df[f'High_{period}d'] = df['High'].rolling(window=period).max()
            df[f'Low_{period}d'] = df['Low'].rolling(window=period).min()
        
        df['High_52w'] = df['High_252d']
        df['Low_52w'] = df['Low_252d']
        df['High_20d'] = df['High_20d']
        df['Low_20d'] = df['Low_20d']
        
        # Distance from MAs
        for period in [10, 20, 50, 100, 200]:
            df[f'Dist_SMA_{period}'] = ((df['Close'] - df[f'SMA_{period}']) / df[f'SMA_{period}']) * 100
            df[f'Dist_EMA_{period}'] = ((df['Close'] - df[f'EMA_{period}']) / df[f'EMA_{period}']) * 100
        
        # MA Slopes
        for period in [10, 20, 50, 200]:
            df[f'SMA_{period}_Slope'] = df[f'SMA_{period}'].diff(5)
            df[f'SMA_{period}_Slope_Pct'] = (df[f'SMA_{period}'].diff(5) / df[f'SMA_{period}']) * 100
        
        # Pivot Points
        df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low'].shift(1)
        df['S1'] = 2 * df['Pivot'] - df['High'].shift(1)
        df['R2'] = df['Pivot'] + (df['High'].shift(1) - df['Low'].shift(1))
        df['S2'] = df['Pivot'] - (df['High'].shift(1) - df['Low'].shift(1))
        df['R3'] = df['High'].shift(1) + 2 * (df['Pivot'] - df['Low'].shift(1))
        df['S3'] = df['Low'].shift(1) - 2 * (df['High'].shift(1) - df['Pivot'])
        
        # Fibonacci Retracement Levels (Multiple periods)
        for period in [20, 50, 100]:
            period_high = df['High'].rolling(window=period).max()
            period_low = df['Low'].rolling(window=period).min()
            diff = period_high - period_low
            df[f'Fib_{period}_236'] = period_high - 0.236 * diff
            df[f'Fib_{period}_382'] = period_high - 0.382 * diff
            df[f'Fib_{period}_500'] = period_high - 0.500 * diff
            df[f'Fib_{period}_618'] = period_high - 0.618 * diff
            df[f'Fib_{period}_786'] = period_high - 0.786 * diff
        
        # Aroon Indicator
        df = self._calculate_aroon(df)
        
        # Ultimate Oscillator
        df = self._calculate_ultimate_oscillator(df)
        
        # Chaikin Money Flow
        df = self._calculate_cmf(df)
        
        # Accumulation/Distribution Line
        df['AD_Line'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
        df['AD_Line'] = df['AD_Line'].cumsum()
        
        # Chaikin Oscillator
        df['AD_EMA_3'] = df['AD_Line'].ewm(span=3, adjust=False).mean()
        df['AD_EMA_10'] = df['AD_Line'].ewm(span=10, adjust=False).mean()
        df['Chaikin_Osc'] = df['AD_EMA_3'] - df['AD_EMA_10']
        
        # Linear Regression
        df = self._calculate_linear_regression(df)
        
        # Correlation with Volume
        df['Price_Volume_Corr'] = df['Close'].rolling(20).corr(df['Volume'])
        
        # Standard Deviation Channels
        for period in [20, 50]:
            linear_reg = df[f'LR_{period}']
            std_dev = df['Close'].rolling(window=period).std()
            df[f'StdDev_{period}_Upper'] = linear_reg + (std_dev * 2)
            df[f'StdDev_{period}_Lower'] = linear_reg - (std_dev * 2)
        
        self.data = df
        print("✅ All 50+ indicators calculated")
        return df
    
    def print_summary(self):
        """Print a summary of the analysis"""
        print("\n" + "=" * 80)
        print(f"📊 ANALYSIS SUMMARY FOR {self.symbol}")
        print("=" * 80)

    def upload_to_gcp(self):
        """Upload analysis files to Google Cloud Storage"""
        print(f"\n☁️ Uploading to GCP Bucket: {self.gcp_bucket}...")
        
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.gcp_bucket)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            date_folder = datetime.now().strftime('%Y-%m-%d')
            current = self.data.iloc[-1]
            
            # 1. SIGNALS FILE (signals_SYMBOL_timestamp.json)
            signals_data = {
                'symbol': self.symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'date': date_folder,
                'price': float(current['Close']),
                'change_pct': float(current.get('Price_Change', 0)),
                'volume': int(current['Volume']),
                'indicators': {
                    'RSI': float(current['RSI']),
                    'MACD': float(current['MACD']),
                    'ADX': float(current.get('ADX', 0)),
                    'Stochastic': float(current['Stoch_K']),
                    'CCI': float(current.get('CCI', 0)),
                    'MFI': float(current.get('MFI', 50)),
                    'BB_Position': float(current.get('BB_Position', 0.5)),
                    'Volatility': float(current.get('Volatility', 20))
                },
                'moving_averages': {
                    'SMA_10': float(current['SMA_10']),
                    'SMA_20': float(current['SMA_20']),
                    'SMA_50': float(current['SMA_50']),
                    'SMA_200': float(current.get('SMA_200', 0)) if not pd.isna(current.get('SMA_200')) else None,
                    'EMA_10': float(current['EMA_10']),
                    'EMA_20': float(current['EMA_20'])
                },
                'signals': self.signals,
                'signal_count': len(self.signals),
                'bullish_count': sum(1 for s in self.signals if 'BULLISH' in s['strength']),
                'bearish_count': sum(1 for s in self.signals if 'BEARISH' in s['strength'])
            }
            
            blob = bucket.blob(f'daily/{date_folder}/signals_{self.symbol}_{timestamp}.json')
            blob.upload_from_string(json.dumps(signals_data, indent=2, default=str), content_type='application/json')
            print(f"  ✅ Uploaded: signals_{self.symbol}_{timestamp}.json")
            
            # 2. COMPLETE ANALYSIS (for your own use)
            complete_analysis = self.generate_comprehensive_analysis()
            if hasattr(self, 'ai_analysis') and self.ai_analysis:
                complete_analysis['ai_analysis'] = self.ai_analysis
            
            blob = bucket.blob(f'daily/{date_folder}/{self.symbol}_complete_analysis_{timestamp}.json')
            blob.upload_from_string(json.dumps(complete_analysis, indent=2, default=str), content_type='application/json')
            print(f"  ✅ Uploaded: {self.symbol}_complete_analysis_{timestamp}.json")
            
            # 3. Technical data CSV
            csv_data = self.data.to_csv()
            blob = bucket.blob(f'daily/{date_folder}/{self.symbol}_technical_data_{timestamp}.csv')
            blob.upload_from_string(csv_data, content_type='text/csv')
            print(f"  ✅ Uploaded: {self.symbol}_technical_data_{timestamp}.csv")
            
            print(f"\n✅ GCP upload complete: gs://{self.gcp_bucket}/daily/{date_folder}/")
            print(f"   Dashboard will show: {self.symbol}")
            return True
            
        except Exception as e:
            print(f"❌ GCP Upload Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def analyze_with_gemini(self):
        """Analyze signals with Gemini AI for deeper insights"""
        print("\n🤖 Analyzing with Gemini AI...")
        
        if not self.genai_client:
            print("   ⚠️ Gemini client not initialized. Skipping AI analysis.")
            return None

        if not self.signals:
            print("   ⚠️ No signals to analyze. Skipping AI analysis.")
            return None

        try:
            # Prepare the prompt
            prompt = self._construct_gemini_prompt()
            
            # Call Gemini API
            model = self.genai_client.get_model('gemini-1.5-flash') # Or 'gemini-pro'
            response = model.generate_content(prompt)
            
            # Process the response
            ai_response = json.loads(response.text)
            
            # Store the analysis
            self.ai_analysis = ai_response
            print("   ✅ Gemini analysis complete.")
            return ai_response

        except Exception as e:
            print(f"   ❌ Gemini AI Error: {str(e)}")
            return None
        
        if self.data is None or self.data.empty:
            print("No data available. Run fetch_data() first.")
            return
            
        current = self.data.iloc[-1]
        print(f"  - Last Close Price: ${current['Close']:.2f}")
        print(f"  - Last Volume: {current['Volume']:,}")
        print(f"  - 52-Week High: ${self.data['High_52w'].iloc[-1]:.2f}")
        print(f"  - 52-Week Low:  ${self.data['Low_52w'].iloc[-1]:.2f}")
        
        if self.signals:
            print(f"\n  - Signals Detected: {len(self.signals)}")
            print(f"    - Bullish: {sum(1 for s in self.signals if 'BULLISH' in s['strength'])}")
            print(f"    - Bearish: {sum(1 for s in self.signals if 'BEARISH' in s['strength'])}")
            print(f"\n  - Top 3 Signals:")
            for i, sig in enumerate(self.signals[:3], 1):
                print(f"    {i}. {sig['signal']} ({sig['strength']}) - Score: {sig.get('ai_score', 'N/A')}")
        else:
            print("\n  - No signals detected.")
            
        print("=" * 80)
    
    def _calculate_atr(self, df, period):
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    def _calculate_adx(self, df, period=14):
        """Calculate ADX and DI"""
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        tr14 = true_range.rolling(period).sum()
        plus_di = 100 * (plus_dm.rolling(period).sum() / tr14)
        minus_di = 100 * (minus_dm.rolling(period).sum() / tr14)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        df['ADX'] = adx
        df['Plus_DI'] = plus_di
        df['Minus_DI'] = minus_di
        
        return df
    
    def _calculate_sar(self, df, af_start=0.02, af_max=0.2):
        """Calculate Parabolic SAR"""
        sar = df['Close'].copy()
        ep = df['Close'].copy()
        af = af_start
        trend = 1
        
        for i in range(1, len(df)):
            if trend == 1:
                sar.iloc[i] = sar.iloc[i-1] + af * (ep.iloc[i-1] - sar.iloc[i-1])
                if df['Low'].iloc[i] < sar.iloc[i]:
                    trend = -1
                    sar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = df['Low'].iloc[i]
                    af = af_start
                else:
                    if df['High'].iloc[i] > ep.iloc[i-1]:
                        ep.iloc[i] = df['High'].iloc[i]
                        af = min(af + af_start, af_max)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
            else:
                sar.iloc[i] = sar.iloc[i-1] - af * (sar.iloc[i-1] - ep.iloc[i-1])
                if df['High'].iloc[i] > sar.iloc[i]:
                    trend = 1
                    sar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = df['High'].iloc[i]
                    af = af_start
                else:
                    if df['Low'].iloc[i] < ep.iloc[i-1]:
                        ep.iloc[i] = df['Low'].iloc[i]
                        af = min(af + af_start, af_max)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
        
        df['SAR'] = sar
        return df
    
    def _calculate_aroon(self, df, period=25):
        """Calculate Aroon Indicator"""
        aroon_up = df['High'].rolling(window=period + 1).apply(
            lambda x: (period - x.argmax()) / period * 100
        )
        aroon_down = df['Low'].rolling(window=period + 1).apply(
            lambda x: (period - x.argmin()) / period * 100
        )
        df['Aroon_Up'] = aroon_up
        df['Aroon_Down'] = aroon_down
        df['Aroon_Osc'] = aroon_up - aroon_down
        return df
    
    def _calculate_ultimate_oscillator(self, df):
        """Calculate Ultimate Oscillator"""
        high_low = df['High'] - df['Low']
        close_prev_close = df['Close'] - df['Close'].shift(1)
        true_range = pd.concat([high_low, close_prev_close.abs()], axis=1).max(axis=1)
        
        buying_pressure = df['Close'] - pd.concat([df['Low'], df['Close'].shift(1)], axis=1).min(axis=1)
        
        avg7 = buying_pressure.rolling(7).sum() / true_range.rolling(7).sum()
        avg14 = buying_pressure.rolling(14).sum() / true_range.rolling(14).sum()
        avg28 = buying_pressure.rolling(28).sum() / true_range.rolling(28).sum()
        
        df['Ultimate_Osc'] = 100 * ((4 * avg7 + 2 * avg14 + avg28) / 7)
        return df
    
    def _calculate_cmf(self, df, period=20):
        """Calculate Chaikin Money Flow"""
        mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        mfv = mfm * df['Volume']
        df['CMF'] = mfv.rolling(period).sum() / df['Volume'].rolling(period).sum()
        return df
    
    def _calculate_linear_regression(self, df):
        """Calculate Linear Regression lines"""
        for period in [20, 50]:
            x = np.arange(period)
            slopes = []
            intercepts = []
            
            for i in range(period - 1, len(df)):
                y = df['Close'].iloc[i - period + 1:i + 1].values
                if len(y) == period:
                    slope, intercept = np.polyfit(x, y, 1)
                    slopes.append(slope * period + intercept)
                    intercepts.append(intercept)
                else:
                    slopes.append(np.nan)
                    intercepts.append(np.nan)
            
            df[f'LR_{period}'] = [np.nan] * (period - 1) + slopes
        
        return df
    
    def detect_signals(self):
        """Detect 200+ comprehensive technical signals"""
        df = self.data.copy()
        current = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3] if len(df) > 2 else prev
        prev5 = df.iloc[-6] if len(df) > 5 else prev
        
        signals = []
        
        print("\n🎯 Scanning for 200+ Technical Alerts...")
        
        # ============ MOVING AVERAGE SIGNALS (40 signals) ============
        
        # Golden/Death Cross
        if len(df) > 200:
            if prev['SMA_50'] <= prev['SMA_200'] and current['SMA_50'] > current['SMA_200']:
                signals.append({
                    'signal': 'GOLDEN CROSS', 
                    'desc': '50 MA crossed above 200 MA', 
                    'strength': 'STRONG BULLISH', 
                    'category': 'MA_CROSS',
                    'value': float(current['SMA_50'])
                })
            
            if prev['SMA_50'] >= prev['SMA_200'] and current['SMA_50'] < current['SMA_200']:
                signals.append({
                    'signal': 'DEATH CROSS', 
                    'desc': '50 MA crossed below 200 MA', 
                    'strength': 'STRONG BEARISH', 
                    'category': 'MA_CROSS',
                    'value': float(current['SMA_50'])
                })
        
        # Short-term MA crosses
        for fast, slow in [(5, 10), (10, 20), (20, 50), (50, 100)]:
            if prev[f'SMA_{fast}'] <= prev[f'SMA_{slow}'] and current[f'SMA_{fast}'] > current[f'SMA_{slow}']:
                signals.append({
                    'signal': f'{fast}/{slow} MA BULL CROSS', 
                    'desc': f'{fast} MA crossed above {slow} MA', 
                    'strength': 'BULLISH', 
                    'category': 'MA_CROSS',
                    'value': float(current[f'SMA_{fast}'])
                })
            
            if prev[f'SMA_{fast}'] >= prev[f'SMA_{slow}'] and current[f'SMA_{fast}'] < current[f'SMA_{slow}']:
                signals.append({
                    'signal': f'{fast}/{slow} MA BEAR CROSS', 
                    'desc': f'{fast} MA crossed below {slow} MA', 
                    'strength': 'BEARISH', 
                    'category': 'MA_CROSS',
                    'value': float(current[f'SMA_{fast}'])
                })
        
        # EMA crosses
        for fast, slow in [(8, 21), (10, 20), (20, 50)]:
            if prev[f'EMA_{fast}'] <= prev[f'EMA_{slow}'] and current[f'EMA_{fast}'] > current[f'EMA_{slow}']:
                signals.append({
                    'signal': f'{fast}/{slow} EMA BULL CROSS', 
                    'desc': f'{fast} EMA crossed above {slow} EMA', 
                    'strength': 'BULLISH', 
                    'category': 'EMA_CROSS',
                    'value': float(current[f'EMA_{fast}'])
                })
        
        # Price vs MA crosses
        for period in [10, 20, 50, 200]:
            if prev['Close'] <= prev[f'SMA_{period}'] and current['Close'] > current[f'SMA_{period}']:
                signals.append({
                    'signal': f'PRICE ABOVE {period}MA', 
                    'desc': f'Price crossed above {period}-day MA', 
                    'strength': 'BULLISH', 
                    'category': 'PRICE_MA_CROSS',
                    'value': float(current[f'SMA_{period}'])
                })
        
        # MA Alignment
        if current['SMA_10'] > current['SMA_20'] > current['SMA_50'] > current.get('SMA_200', 0):
            signals.append({
                'signal': 'PERFECT MA ALIGNMENT', 
                'desc': '10>20>50>200 MA alignment', 
                'strength': 'EXTREME BULLISH', 
                'category': 'MA_ALIGNMENT',
                'value': 100
            })
        
        # MA Compression
        ma_range = (current['SMA_50'] - current['SMA_10']) / current['SMA_50'] * 100
        if abs(ma_range) < 2:
            signals.append({
                'signal': 'MA COMPRESSION', 
                'desc': 'MAs converging - breakout imminent', 
                'strength': 'NEUTRAL', 
                'category': 'MA_COMPRESSION',
                'value': abs(ma_range)
            })
        
        # MA Slope Analysis
        for period in [10, 20, 50]:
            slope_pct = current.get(f'SMA_{period}_Slope_Pct', 0)
            if slope_pct > 1:
                signals.append({
                    'signal': f'{period}MA STEEP UPTREND', 
                    'desc': f'{period}MA slope: +{slope_pct:.2f}%', 
                    'strength': 'BULLISH', 
                    'category': 'MA_SLOPE',
                    'value': slope_pct
                })
            elif slope_pct < -1:
                signals.append({
                    'signal': f'{period}MA STEEP DOWNTREND', 
                    'desc': f'{period}MA slope: {slope_pct:.2f}%', 
                    'strength': 'BEARISH', 
                    'category': 'MA_SLOPE',
                    'value': slope_pct
                })
        
        # Distance from MAs
        for period in [20, 50, 200]:
            dist = current.get(f'Dist_SMA_{period}', 0)
            if dist > 15:
                signals.append({
                    'signal': f'OVEREXTENDED ABOVE {period}MA', 
                    'desc': f'{dist:.1f}% above {period}MA', 
                    'strength': 'BEARISH',
                    'category': 'MA_DISTANCE',
                    'value': dist
                })
            elif dist < -15:
                signals.append({
                    'signal': f'OVEREXTENDED BELOW {period}MA', 
                    'desc': f'{abs(dist):.1f}% below {period}MA', 
                    'strength': 'BULLISH',
                    'category': 'MA_DISTANCE',
                    'value': dist
                })
        
        # ============ RSI SIGNALS (30 signals) ============
        
        # RSI levels (multiple periods)
        for period in [9, 14, 21]:
            rsi = current.get(f'RSI_{period}', 50)
            if rsi < 20:
                signals.append({
                    'signal': f'RSI{period} EXTREME OVERSOLD',
                    'desc': f'RSI({period}): {rsi:.1f}',
                    'strength': 'EXTREME BULLISH',
                    'category': 'RSI',
                    'value': rsi
                })
            elif rsi < 30:
                signals.append({
                    'signal': f'RSI{period} OVERSOLD',
                    'desc': f'RSI({period}): {rsi:.1f}',
                    'strength': 'BULLISH',
                    'category': 'RSI',
                    'value': rsi
                })
            elif rsi > 80:
                signals.append({
                    'signal': f'RSI{period} EXTREME OVERBOUGHT',
                    'desc': f'RSI({period}): {rsi:.1f}',
                    'strength': 'EXTREME BEARISH',
                    'category': 'RSI',
                    'value': rsi
                })
            elif rsi > 70:
                signals.append({
                    'signal': f'RSI{period} OVERBOUGHT',
                    'desc': f'RSI({period}): {rsi:.1f}',
                    'strength': 'BEARISH',
                    'category': 'RSI',
                    'value': rsi
                })
        
        # RSI Divergences
        if len(df) > 20:
            price_change = current['Close'] - df['Close'].iloc[-20]
            rsi_change = current['RSI'] - df['RSI'].iloc[-20]
            
            if price_change < 0 and rsi_change > 0:
                signals.append({
                    'signal': 'RSI BULLISH DIVERGENCE',
                    'desc': 'Price declining but RSI rising',
                    'strength': 'STRONG BULLISH',
                    'category': 'RSI_DIVERGENCE',
                    'value': rsi_change
                })
            elif price_change > 0 and rsi_change < 0:
                signals.append({
                    'signal': 'RSI BEARISH DIVERGENCE',
                    'desc': 'Price rising but RSI falling',
                    'strength': 'STRONG BEARISH',
                    'category': 'RSI_DIVERGENCE',
                    'value': rsi_change
                })
        
        # RSI Momentum
        if len(df) > 5:
            rsi_momentum = current['RSI'] - prev5['RSI']
            if rsi_momentum > 15:
                signals.append({
                    'signal': 'RSI MOMENTUM SURGE',
                    'desc': f'RSI +{rsi_momentum:.1f} in 5 days',
                    'strength': 'STRONG BULLISH',
                    'category': 'RSI_MOMENTUM',
                    'value': rsi_momentum
                })
            elif rsi_momentum < -15:
                signals.append({
                    'signal': 'RSI MOMENTUM COLLAPSE',
                    'desc': f'RSI {rsi_momentum:.1f} in 5 days',
                    'strength': 'STRONG BEARISH',
                    'category': 'RSI_MOMENTUM',
                    'value': rsi_momentum
                })
        
        # ============ MACD SIGNALS (25 signals) ============
        
        # MACD Crosses
        if prev['MACD'] <= prev['MACD_Signal'] and current['MACD'] > current['MACD_Signal']:
            signals.append({
                'signal': 'MACD BULL CROSS',
                'desc': 'MACD crossed above signal',
                'strength': 'STRONG BULLISH',
                'category': 'MACD',
                'value': float(current['MACD'])
            })
        
        if prev['MACD'] >= prev['MACD_Signal'] and current['MACD'] < current['MACD_Signal']:
            signals.append({
                'signal': 'MACD BEAR CROSS',
                'desc': 'MACD crossed below signal',
                'strength': 'STRONG BEARISH',
                'category': 'MACD',
                'value': float(current['MACD'])
            })
        
        # Fast MACD
        if prev['MACD_Fast'] <= prev['MACD_Fast_Signal'] and current['MACD_Fast'] > current['MACD_Fast_Signal']:
            signals.append({
                'signal': 'FAST MACD BULL CROSS',
                'desc': 'Fast MACD bullish signal',
                'strength': 'BULLISH',
                'category': 'MACD',
                'value': float(current['MACD_Fast'])
            })
        
        # MACD Histogram Expansion
        if abs(current['MACD_Hist']) > abs(prev['MACD_Hist']) * 1.2:
            direction = 'BULLISH' if current['MACD_Hist'] > 0 else 'BEARISH'
            signals.append({
                'signal': f'MACD HISTOGRAM EXPANSION {direction}',
                'desc': 'Momentum accelerating',
                'strength': f'STRONG {direction}',
                'category': 'MACD_MOMENTUM',
                'value': float(current['MACD_Hist'])
            })
        
        # MACD Zero Line
        if prev['MACD'] <= 0 and current['MACD'] > 0:
            signals.append({
                'signal': 'MACD ABOVE ZERO',
                'desc': 'MACD crossed into positive territory',
                'strength': 'BULLISH',
                'category': 'MACD',
                'value': float(current['MACD'])
            })
        
        # ============ BOLLINGER BANDS (25 signals) ============
        
        # BB Squeeze
        for period in [10, 20, 30]:
            bb_width = current.get(f'BB_{period}_Width', 0)
            bb_avg = df[f'BB_{period}_Width'].tail(50).mean()
            if bb_width < bb_avg * 0.6:
                signals.append({
                    'signal': f'BB{period} SQUEEZE',
                    'desc': f'{period}-period bands narrowing',
                    'strength': 'NEUTRAL',
                    'category': 'BB_SQUEEZE',
                    'value': bb_width
                })
        
        # BB Position
        bb_pos = current.get('BB_Position', 0.5)
        if bb_pos > 0.95:
            signals.append({
                'signal': 'AT UPPER BB',
                'desc': 'Price in top 5% of BB range',
                'strength': 'BEARISH',
                'category': 'BOLLINGER',
                'value': bb_pos * 100
            })
        elif bb_pos < 0.05:
            signals.append({
                'signal': 'AT LOWER BB',
                'desc': 'Price in bottom 5% of BB range',
                'strength': 'BULLISH',
                'category': 'BOLLINGER',
                'value': bb_pos * 100
            })
        
        # BB Breakouts
        if current['Close'] > current['BB_Upper']:
            signals.append({
                'signal': 'ABOVE UPPER BB',
                'desc': 'Price broke above upper band',
                'strength': 'EXTREME BULLISH',
                'category': 'BB_BREAKOUT',
                'value': float(current['Close'] - current['BB_Upper'])
            })
        
        if current['Close'] < current['BB_Lower']:
            signals.append({
                'signal': 'BELOW LOWER BB',
                'desc': 'Price broke below lower band',
                'strength': 'EXTREME BEARISH',
                'category': 'BB_BREAKOUT',
                'value': float(current['BB_Lower'] - current['Close'])
            })
        
        # BB Walk
        if len(df) > 5:
            bb_walk_bull = all(df['Close'].iloc[-i] >= df['BB_Upper'].iloc[-i] * 0.98 for i in range(1, 5))
            if bb_walk_bull:
                signals.append({
                    'signal': 'BB WALK UPPER',
                    'desc': 'Price riding upper band - strong trend',
                    'strength': 'EXTREME BULLISH',
                    'category': 'BB_WALK',
                    'value': 100
                })
        
        # ============ VOLUME SIGNALS (25 signals) ============
        
        # Volume Spikes
        vol_ratio = current['Volume'] / current['Volume_MA_20']
        if vol_ratio > 3:
            signals.append({
                'signal': 'EXTREME VOLUME 3X',
                'desc': f'Volume: {vol_ratio:.1f}x average',
                'strength': 'VERY SIGNIFICANT',
                'category': 'VOLUME',
                'value': vol_ratio
            })
        elif vol_ratio > 2:
            signals.append({
                'signal': 'VOLUME SPIKE 2X',
                'desc': f'Volume: {vol_ratio:.1f}x average',
                'strength': 'SIGNIFICANT',
                'category': 'VOLUME',
                'value': vol_ratio
            })
        
        # Volume with Price Action
        if current['Price_Change'] > 3 and vol_ratio > 1.5:
            signals.append({
                'signal': 'VOLUME BREAKOUT',
                'desc': 'High volume + strong price gain',
                'strength': 'EXTREME BULLISH',
                'category': 'VOLUME_BREAKOUT',
                'value': current['Price_Change']
            })
        
        if current['Price_Change'] < -3 and vol_ratio > 1.5:
            signals.append({
                'signal': 'VOLUME SELLOFF',
                'desc': 'High volume + strong price decline',
                'strength': 'EXTREME BEARISH',
                'category': 'VOLUME_SELLOFF',
                'value': current['Price_Change']
            })
        
        # Volume Climax
        if vol_ratio > 4:
            direction = 'BUYING' if current['Close'] > current['Open'] else 'SELLING'
            signals.append({
                'signal': f'{direction} CLIMAX',
                'desc': 'Extreme volume event',
                'strength': 'EXTREME',
                'category': 'VOLUME_CLIMAX',
                'value': vol_ratio
            })
        
        # OBV Trend
        if len(df) > 20:
            obv_change = (current['OBV'] - df['OBV'].iloc[-20]) / abs(df['OBV'].iloc[-20]) * 100
            if obv_change > 30:
                signals.append({
                    'signal': 'OBV STRONG ACCUMULATION',
                    'desc': f'OBV +{obv_change:.0f}% in 20 days',
                    'strength': 'STRONG BULLISH',
                    'category': 'OBV',
                    'value': obv_change
                })
            elif obv_change < -30:
                signals.append({
                    'signal': 'OBV STRONG DISTRIBUTION',
                    'desc': f'OBV {obv_change:.0f}% in 20 days',
                    'strength': 'STRONG BEARISH',
                    'category': 'OBV',
                    'value': obv_change
                })
        
        # Chaikin Money Flow
        cmf = current.get('CMF', 0)
        if cmf > 0.25:
            signals.append({
                'signal': 'CMF STRONG BUYING',
                'desc': f'CMF: {cmf:.2f} - strong accumulation',
                'strength': 'BULLISH',
                'category': 'CMF',
                'value': cmf
            })
        elif cmf < -0.25:
            signals.append({
                'signal': 'CMF STRONG SELLING',
                'desc': f'CMF: {cmf:.2f} - strong distribution',
                'strength': 'BEARISH',
                'category': 'CMF',
                'value': cmf
            })
        
        # ============ STOCHASTIC & OSCILLATORS (20 signals) ============
        
        # Stochastic
        if current['Stoch_K'] < 20:
            signals.append({
                'signal': 'STOCHASTIC OVERSOLD',
                'desc': f'%K: {current["Stoch_K"]:.1f}',
                'strength': 'BULLISH',
                'category': 'STOCHASTIC',
                'value': current['Stoch_K']
            })
        elif current['Stoch_K'] > 80:
            signals.append({
                'signal': 'STOCHASTIC OVERBOUGHT',
                'desc': f'%K: {current["Stoch_K"]:.1f}',
                'strength': 'BEARISH',
                'category': 'STOCHASTIC',
                'value': current['Stoch_K']
            })
        
        # Stochastic Cross
        if prev['Stoch_K'] <= prev['Stoch_D'] and current['Stoch_K'] > current['Stoch_D']:
            signals.append({
                'signal': 'STOCHASTIC BULL CROSS',
                'desc': '%K crossed above %D',
                'strength': 'BULLISH',
                'category': 'STOCHASTIC',
                'value': current['Stoch_K']
            })
        
        # Williams %R
        williams = current.get('Williams_R', -50)
        if williams < -80:
            signals.append({
                'signal': 'WILLIAMS R OVERSOLD',
                'desc': f'W%R: {williams:.1f}',
                'strength': 'BULLISH',
                'category': 'WILLIAMS_R',
                'value': williams
            })
        elif williams > -20:
            signals.append({
                'signal': 'WILLIAMS R OVERBOUGHT',
                'desc': f'W%R: {williams:.1f}',
                'strength': 'BEARISH',
                'category': 'WILLIAMS_R',
                'value': williams
            })
        
        # CCI
        cci = current.get('CCI', 0)
        if cci > 200:
            signals.append({
                'signal': 'CCI EXTREME OVERBOUGHT',
                'desc': f'CCI: {cci:.1f}',
                'strength': 'EXTREME BEARISH',
                'category': 'CCI',
                'value': cci
            })
        elif cci < -200:
            signals.append({
                'signal': 'CCI EXTREME OVERSOLD',
                'desc': f'CCI: {cci:.1f}',
                'strength': 'EXTREME BULLISH',
                'category': 'CCI',
                'value': cci
            })
        
        # MFI
        mfi = current.get('MFI', 50)
        if mfi < 20:
            signals.append({
                'signal': 'MFI OVERSOLD',
                'desc': f'Money Flow: {mfi:.1f}',
                'strength': 'BULLISH',
                'category': 'MFI',
                'value': mfi
            })
        elif mfi > 80:
            signals.append({
                'signal': 'MFI OVERBOUGHT',
                'desc': f'Money Flow: {mfi:.1f}',
                'strength': 'BEARISH',
                'category': 'MFI',
                'value': mfi
            })
        
        # Ultimate Oscillator
        uo = current.get('Ultimate_Osc', 50)
        if uo < 30:
            signals.append({
                'signal': 'ULTIMATE OSC OVERSOLD',
                'desc': f'UO: {uo:.1f}',
                'strength': 'BULLISH',
                'category': 'ULTIMATE_OSC',
                'value': uo
            })
        elif uo > 70:
            signals.append({
                'signal': 'ULTIMATE OSC OVERBOUGHT',
                'desc': f'UO: {uo:.1f}',
                'strength': 'BEARISH',
                'category': 'ULTIMATE_OSC',
                'value': uo
            })
        
        # ============ TREND SIGNALS (20 signals) ============
        
        # ADX Trend Strength
        adx = current.get('ADX', 0)
        if adx > 40:
            direction = 'UP' if current['Plus_DI'] > current['Minus_DI'] else 'DOWN'
            signals.append({
                'signal': f'VERY STRONG {direction}TREND',
                'desc': f'ADX: {adx:.1f}',
                'strength': 'EXTREME',
                'category': 'ADX',
                'value': adx
            })
        elif adx > 25:
            direction = 'UP' if current['Plus_DI'] > current['Minus_DI'] else 'DOWN'
            signals.append({
                'signal': f'STRONG {direction}TREND',
                'desc': f'ADX: {adx:.1f}',
                'strength': 'TRENDING',
                'category': 'ADX',
                'value': adx
            })
        
        # DI Crossover
        if prev['Plus_DI'] <= prev['Minus_DI'] and current['Plus_DI'] > current['Minus_DI']:
            signals.append({
                'signal': 'DI BULL CROSS',
                'desc': '+DI crossed above -DI',
                'strength': 'BULLISH',
                'category': 'ADX',
                'value': current['Plus_DI'] - current['Minus_DI']
            })
        
        # Aroon
        aroon_osc = current.get('Aroon_Osc', 0)
        if aroon_osc > 70:
            signals.append({
                'signal': 'AROON STRONG UPTREND',
                'desc': f'Aroon Osc: {aroon_osc:.1f}',
                'strength': 'STRONG BULLISH',
                'category': 'AROON',
                'value': aroon_osc
            })
        elif aroon_osc < -70:
            signals.append({
                'signal': 'AROON STRONG DOWNTREND',
                'desc': f'Aroon Osc: {aroon_osc:.1f}',
                'strength': 'STRONG BEARISH',
                'category': 'AROON',
                'value': aroon_osc
            })
        
        # Parabolic SAR
        if current['Close'] > current.get('SAR', 0) and prev['Close'] <= prev.get('SAR', 0):
            signals.append({
                'signal': 'SAR BULL SIGNAL',
                'desc': 'Price crossed above SAR',
                'strength': 'BULLISH',
                'category': 'SAR',
                'value': float(current['Close'] - current.get('SAR', 0))
            })
        
        # ============ PRICE ACTION (20 signals) ============
        
        # Large moves
        price_change = current.get('Price_Change', 0)
        if price_change > 10:
            signals.append({
                'signal': 'EXPLOSIVE MOVE UP',
                'desc': f'+{price_change:.1f}% today',
                'strength': 'EXTREME BULLISH',
                'category': 'PRICE_ACTION',
                'value': price_change
            })
        elif price_change > 5:
            signals.append({
                'signal': 'LARGE GAIN',
                'desc': f'+{price_change:.1f}% today',
                'strength': 'STRONG BULLISH',
                'category': 'PRICE_ACTION',
                'value': price_change
            })
        elif price_change < -10:
            signals.append({
                'signal': 'EXPLOSIVE MOVE DOWN',
                'desc': f'{price_change:.1f}% today',
                'strength': 'EXTREME BEARISH',
                'category': 'PRICE_ACTION',
                'value': price_change
            })
        elif price_change < -5:
            signals.append({
                'signal': 'LARGE LOSS',
                'desc': f'{price_change:.1f}% today',
                'strength': 'STRONG BEARISH',
                'category': 'PRICE_ACTION',
                'value': price_change
            })
        
        # Gaps
        gap = (current['Open'] - prev['Close']) / prev['Close'] * 100
        if gap > 3:
            signals.append({
                'signal': 'LARGE GAP UP',
                'desc': f'Opened {gap:.1f}% higher',
                'strength': 'STRONG BULLISH',
                'category': 'GAP',
                'value': gap
            })
        elif gap < -3:
            signals.append({
                'signal': 'LARGE GAP DOWN',
                'desc': f'Opened {abs(gap):.1f}% lower',
                'strength': 'STRONG BEARISH',
                'category': 'GAP',
                'value': gap
            })
        
        # Candlestick Patterns
        body = abs(current['Close'] - current['Open'])
        candle_range = current['High'] - current['Low']
        upper_wick = current['High'] - max(current['Open'], current['Close'])
        lower_wick = min(current['Open'], current['Close']) - current['Low']
        
        # Doji
        if candle_range > 0 and body / candle_range < 0.1:
            signals.append({
                'signal': 'DOJI CANDLE',
                'desc': 'Indecision pattern',
                'strength': 'NEUTRAL',
                'category': 'CANDLESTICK',
                'value': body / candle_range * 100
            })
        
        # Hammer
        if lower_wick > 2 * body and upper_wick < body and current['Close'] < prev['Close']:
            signals.append({
                'signal': 'HAMMER PATTERN',
                'desc': 'Potential bullish reversal',
                'strength': 'BULLISH',
                'category': 'CANDLESTICK',
                'value': lower_wick / body
            })
        
        # Shooting Star
        if upper_wick > 2 * body and lower_wick < body and current['Close'] > prev['Close']:
            signals.append({
                'signal': 'SHOOTING STAR',
                'desc': 'Potential bearish reversal',
                'strength': 'BEARISH',
                'category': 'CANDLESTICK',
                'value': upper_wick / body
            })
        
        # Engulfing
        if current['Close'] > current['Open'] and prev['Close'] < prev['Open']:
            if current['Open'] <= prev['Close'] and current['Close'] >= prev['Open']:
                signals.append({
                    'signal': 'BULLISH ENGULFING',
                    'desc': 'Strong reversal pattern',
                    'strength': 'STRONG BULLISH',
                    'category': 'CANDLESTICK',
                    'value': body
                })
        
        # ============ 52-WEEK & RANGE (15 signals) ============
        
        # 52-week levels
        if current['Close'] >= current['High_52w'] * 0.999:
            signals.append({
                'signal': '52-WEEK HIGH',
                'desc': f'At ${current["Close"]:.2f}',
                'strength': 'EXTREME BULLISH',
                'category': 'RANGE',
                'value': float(current['Close'])
            })
        
        if current['Close'] <= current['Low_52w'] * 1.001:
            signals.append({
                'signal': '52-WEEK LOW',
                'desc': f'At ${current["Close"]:.2f}',
                'strength': 'EXTREME BEARISH',
                'category': 'RANGE',
                'value': float(current['Close'])
            })
        
        # 52-week position
        week_52_position = ((current['Close'] - current['Low_52w']) / 
                           (current['High_52w'] - current['Low_52w'])) * 100
        if week_52_position > 95:
            signals.append({
                'signal': 'TOP OF 52W RANGE',
                'desc': f'At {week_52_position:.0f}% of 52w range',
                'strength': 'OVERBOUGHT',
                'category': 'RANGE',
                'value': week_52_position
            })
        elif week_52_position < 5:
            signals.append({
                'signal': 'BOTTOM OF 52W RANGE',
                'desc': f'At {week_52_position:.0f}% of 52w range',
                'strength': 'OVERSOLD',
                'category': 'RANGE',
                'value': week_52_position
            })
        
        # ============ ADDITIONAL ADVANCED SIGNALS (20 signals) ============
        
        # Ichimoku Cloud
        if current['Close'] > current['Senkou_A'] and current['Close'] > current['Senkou_B']:
            signals.append({
                'signal': 'ABOVE ICHIMOKU CLOUD',
                'desc': 'Bullish cloud position',
                'strength': 'BULLISH',
                'category': 'ICHIMOKU',
                'value': float(current['Close'] - max(current['Senkou_A'], current['Senkou_B']))
            })
        
        # Tenkan/Kijun Cross
        if prev['Tenkan'] <= prev['Kijun'] and current['Tenkan'] > current['Kijun']:
            signals.append({
                'signal': 'TENKAN/KIJUN BULL CROSS',
                'desc': 'Ichimoku bullish signal',
                'strength': 'BULLISH',
                'category': 'ICHIMOKU',
                'value': float(current['Tenkan'] - current['Kijun'])
            })
        
        # ROC Extremes
        for period in [10, 20]:
            roc = current.get(f'ROC_{period}', 0)
            if roc > 20:
                signals.append({
                    'signal': f'ROC{period} EXTREME POSITIVE',
                    'desc': f'{period}-day ROC: +{roc:.1f}%',
                    'strength': 'STRONG BULLISH',
                    'category': 'ROC',
                    'value': roc
                })
            elif roc < -20:
                signals.append({
                    'signal': f'ROC{period} EXTREME NEGATIVE',
                    'desc': f'{period}-day ROC: {roc:.1f}%',
                    'strength': 'STRONG BEARISH',
                    'category': 'ROC',
                    'value': roc
                })
        
        # VWAP
        if current['Close'] > current['VWAP'] and prev['Close'] <= prev['VWAP']:
            signals.append({
                'signal': 'ABOVE VWAP',
                'desc': 'Institutional buying support',
                'strength': 'BULLISH',
                'category': 'VWAP',
                'value': float(current['Close'] - current['VWAP'])
            })
        
        # Volatility
        vol = current.get('Volatility', 0)
        if vol > 60:
            signals.append({
                'signal': 'EXTREME VOLATILITY',
                'desc': f'{vol:.0f}% annualized',
                'strength': 'HIGH RISK',
                'category': 'VOLATILITY',
                'value': vol
            })
        elif vol < 15:
            signals.append({
                'signal': 'LOW VOLATILITY',
                'desc': 'Compression - breakout likely',
                'strength': 'NEUTRAL',
                'category': 'VOLATILITY',
                'value': vol
            })
        
        # Pivot Points
        if current['Close'] > current['R1'] and prev['Close'] <= prev['R1']:
            signals.append({
                'signal': 'BROKE R1 RESISTANCE',
                'desc': f'Above ${current["R1"]:.2f}',
                'strength': 'BULLISH',
                'category': 'PIVOT',
                'value': float(current['Close'] - current['R1'])
            })
        
        if current['Close'] < current['S1'] and prev['Close'] >= prev['S1']:
            signals.append({
                'signal': 'BROKE S1 SUPPORT',
                'desc': f'Below ${current["S1"]:.2f}',
                'strength': 'BEARISH',
                'category': 'PIVOT',
                'value': float(current['S1'] - current['Close'])
            })
        
        self.signals = signals
        print(f"✅ Detected {len(signals)} Active Signals")
        return signals
    
    def generate_comprehensive_analysis(self):
        """Generate complete analysis with indicator ratings"""
        current = self.data.iloc[-1]
        
        # Calculate indicator scores (0-100)
        indicator_ratings = {
            'trend_strength': self._rate_trend_strength(current),
            'momentum': self._rate_momentum(current),
            'volatility': self._rate_volatility(current),
            'volume': self._rate_volume(current),
            'support_resistance': self._rate_support_resistance(current),
            'oversold_overbought': self._rate_oversold_overbought(current)
        }
        
        # Calculate overall bias
        bullish_signals = sum(1 for s in self.signals if 'BULLISH' in s['strength'])
        bearish_signals = sum(1 for s in self.signals if 'BEARISH' in s['strength'])
        
        if bullish_signals > bearish_signals * 1.5:
            overall_bias = 'STRONG BULLISH'
            bias_score = 85
        elif bullish_signals > bearish_signals:
            overall_bias = 'BULLISH'
            bias_score = 65
        elif bearish_signals > bullish_signals * 1.5:
            overall_bias = 'STRONG BEARISH'
            bias_score = 15
        elif bearish_signals > bullish_signals:
            overall_bias = 'BEARISH'
            bias_score = 35
        else:
            overall_bias = 'NEUTRAL'
            bias_score = 50
        
        analysis = {
            'symbol': self.symbol,
            'timestamp': datetime.now().isoformat(),
            'price_data': {
                'current_price': float(current['Close']),
                'open': float(current['Open']),
                'high': float(current['High']),
                'low': float(current['Low']),
                'volume': int(current['Volume']),
                'change': float(current.get('Price_Change', 0)),
                'change_5d': float(current.get('Price_Change_5d', 0)),
                'change_10d': float(current.get('Price_Change_10d', 0))
            },
            'indicator_ratings': indicator_ratings,
            'overall_bias': overall_bias,
            'bias_score': bias_score,
            'signal_summary': {
                'total_signals': len(self.signals),
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'neutral_signals': len(self.signals) - bullish_signals - bearish_signals
            },
            'key_levels': self._identify_key_levels(current),
            'signals': self.signals
        }
        
        return analysis
    
    def _rate_trend_strength(self, current):
        """Rate trend strength 0-100"""
        score = 50
        
        # ADX contribution
        adx = current.get('ADX', 0)
        if adx > 40:
            score += 25
        elif adx > 25:
            score += 15
        elif adx < 15:
            score -= 15
        
        # MA alignment
        if current['SMA_10'] > current['SMA_20'] > current['SMA_50']:
            score += 15
        elif current['SMA_10'] < current['SMA_20'] < current['SMA_50']:
            score -= 15
        
        # Price vs 200 MA
        if len(self.data) > 200:
            if current['Close'] > current['SMA_200']:
                score += 10
            else:
                score -= 10
        
        return max(0, min(100, score))
    
    def _rate_momentum(self, current):
        """Rate momentum 0-100"""
        score = 50
        
        # RSI
        rsi = current.get('RSI', 50)
        if rsi > 60:
            score += (rsi - 60) * 0.5
        elif rsi < 40:
            score -= (40 - rsi) * 0.5
        
        # MACD
        if current['MACD'] > current['MACD_Signal']:
            score += 15
        else:
            score -= 15
        
        # ROC
        roc_20 = current.get('ROC_20', 0)
        if roc_20 > 10:
            score += 15
        elif roc_20 < -10:
            score -= 15
        
        return max(0, min(100, score))
    
    def _rate_volatility(self, current):
        """Rate volatility 0-100 (higher = more volatile)"""
        vol = current.get('Volatility', 20)
        
        if vol > 80:
            return 95
        elif vol > 60:
            return 80
        elif vol > 40:
            return 65
        elif vol > 25:
            return 50
        elif vol > 15:
            return 35
        else:
            return 20
    
    def _rate_volume(self, current):
        """Rate volume strength 0-100"""
        vol_ratio = current['Volume'] / current['Volume_MA_20']
        
        if vol_ratio > 2.5:
            score = 95
        elif vol_ratio > 1.5:
            score = 80
        elif vol_ratio > 1.0:
            score = 60
        elif vol_ratio > 0.7:
            score = 40
        else:
            score = 20
        
        # OBV trend
        if len(self.data) > 10:
            obv_trend = (current['OBV'] - self.data['OBV'].iloc[-10]) / abs(self.data['OBV'].iloc[-10])
            if obv_trend > 0.1:
                score += 10
            elif obv_trend < -0.1:
                score -= 10
        
        return max(0, min(100, score))
    
    def _rate_support_resistance(self, current):
        """Rate proximity to support/resistance 0-100"""
        score = 50
        
        # Distance from 52-week high/low
        dist_high = ((current['High_52w'] - current['Close']) / current['High_52w']) * 100
        dist_low = ((current['Close'] - current['Low_52w']) / current['Low_52w']) * 100
        
        if dist_high < 5:
            score += 20
        if dist_low < 5:
            score -= 20
        
        # Pivot points
        if abs(current['Close'] - current['R1']) / current['Close'] < 0.02:
            score += 15
        if abs(current['Close'] - current['S1']) / current['Close'] < 0.02:
            score -= 15
        
        return max(0, min(100, score))
    
    def _rate_oversold_overbought(self, current):
        """Rate oversold/overbought 0-100 (50=neutral)"""
        score = 50
        
        # RSI
        rsi = current.get('RSI', 50)
        score += (rsi - 50) * 0.5
        
        # Stochastic
        stoch = current.get('Stoch_K', 50)
        score += (stoch - 50) * 0.3
        
        # MFI
        mfi = current.get('MFI', 50)
        score += (mfi - 50) * 0.2
        
        return max(0, min(100, score))
    
    def _identify_key_levels(self, current):
        """Identify key support and resistance levels"""
        return {
            'resistance': [
                {'level': float(current['R1']), 'type': 'Pivot R1'},
                {'level': float(current['R2']), 'type': 'Pivot R2'},
                {'level': float(current['High_20d']), 'type': '20-day High'},
                {'level': float(current['High_52w']), 'type': '52-week High'},
                {'level': float(current['BB_Upper']), 'type': 'BB Upper'}
            ],
            'support': [
                {'level': float(current['S1']), 'type': 'Pivot S1'},
                {'level': float(current['S2']), 'type': 'Pivot S2'},
                {'level': float(current['Low_20d']), 'type': '20-day Low'},
                {'level': float(current['Low_52w']), 'type': '52-week Low'},
                {'level': float(current['BB_Lower']), 'type': 'BB Lower'}
            ],
            'moving_averages': {
                'SMA_20': float(current['SMA_20']),
                'SMA_50': float(current['SMA_50']),
                'SMA_200': float(current.get('SMA_200', 0)),
                'EMA_20': float(current['EMA_20'])
            }
        }
    
    def analyze_with_ai(self):
        """Multi-stage AI analysis"""
        if not self.genai_client:
            print("\n⚠️  Gemini API key not provided. Skipping AI analysis.")
            return None
        
        print("\n🤖 Starting comprehensive AI analysis...")
        
        current = self.data.iloc[-1]
        analysis = self.generate_comprehensive_analysis()
        
        # Stage 1: Rank all signals
        print("  📊 Stage 1: Ranking signals...")
        self._rank_signals_with_ai()
        
        # Stage 2: Deep market analysis
        print("  🔍 Stage 2: Deep market analysis...")
        market_analysis = self._get_market_analysis(current, analysis)
        
        # Stage 3: Options strategy recommendations
        print("  💰 Stage 3: Generating trade recommendations...")
        trade_recommendations = self._get_trade_recommendations(current, analysis)
        print(f"📁 trade recs: {trade_recommendations}")
        return {
            'market_analysis': market_analysis,
            'trade_recommendations': trade_recommendations,
            'ranked_signals': self.signals[:20]  # Top 20 signals
            
        }
          
     
    def _rank_signals_with_ai(self):
        """Rank signals with AI scoring"""
        if len(self.signals) == 0:
            return
        
        # Process in batches
        batch_size = 50
        for i in range(0, len(self.signals), batch_size):
            batch = self.signals[i:i+batch_size]
            
            try:
                current = self.data.iloc[-1]
                
                prompt = f"""Score these trading signals for {self.symbol} (Price: ${current['Close']:.2f}).
Score each 1-100 based on: reliability, timing, risk/reward, actionability.

SIGNALS:
"""
                for idx, sig in enumerate(batch, i+1):
                    prompt += f"{idx}. {sig['signal']}: {sig['desc']} [{sig['category']}]\n"
                
                prompt += """
Return ONLY valid JSON:
{"scores":[{"n":1,"score":85,"why":"brief reason"},...]}

Keep reasons under 40 chars."""

                response = self.genai_client.models.generate_content(
                    model='gemini-2.0-flash-exp',
                    contents=prompt
                )
                
                response_text = response.text.strip()
                
                # Clean response
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0].strip()
                
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    response_text = response_text[start_idx:end_idx+1]
                
                scores_data = json.loads(response_text)
                
                for score_item in scores_data.get('scores', []):
                    sig_idx = score_item['n'] - 1
                    if 0 <= sig_idx < len(self.signals):
                        self.signals[sig_idx]['ai_score'] = score_item.get('score', 50)
                        self.signals[sig_idx]['ai_reasoning'] = score_item.get('why', 'N/A')
                
            except Exception as e:
                print(f"    ⚠️  Batch {i//batch_size + 1} scoring error: {str(e)[:50]}")
                for sig in batch:
                    if 'ai_score' not in sig:
                        sig['ai_score'] = 50
                        sig['ai_reasoning'] = 'Scoring error'
        
        # Sort by score
        self.signals.sort(key=lambda x: x.get('ai_score', 0), reverse=True)
        
        # Add ranks
        for rank, signal in enumerate(self.signals, 1):
            signal['rank'] = rank
        
        print(f"  ✅ Ranked {len(self.signals)} signals")
    
    def _get_market_analysis(self, current, analysis):
        """Get comprehensive market analysis from AI"""
        try:
            # Add sector and industry to the prompt if available
            sector = self.ticker_info.get('sector', 'N/A')
            industry = self.ticker_info.get('industry', 'N/A')

            prompt = f"""Create a detailed, article-style investment analysis for {self.symbol}.
The output must be a single, valid JSON object.

COMPANY CONTEXT:
- Sector: {sector}
- Industry: {industry}
- Current Price: ${current['Close']:.2f}
- Recent Change: {current.get('Price_Change', 0):+.2f}%

TECHNICAL OVERVIEW:
- Overall Bias: {analysis['overall_bias']} (Score: {analysis['bias_score']}/100)
- Trend Strength: {analysis['indicator_ratings']['trend_strength']}/100
- Momentum: {analysis['indicator_ratings']['momentum']}/100
- Volume Activity: {analysis['indicator_ratings']['volume']}/100
- Key Levels: Support ${analysis['key_levels']['support'][0]['level']:.2f}, Resistance ${analysis['key_levels']['resistance'][0]['level']:.2f}

TOP TECHNICAL SIGNALS:
{self._format_top_signals_for_article()}

ARTICLE REQUIREMENTS:
Return a single, valid JSON object with the following structure.

{{
  "headline": "Compelling, 8-12 word headline about the stock's current situation.",
  "key_points": [
    "Key takeaway 1: A crucial, concise insight.",
    "Key takeaway 2: Another important point.",
    "Key takeaway 3: A final summary point."
  ],
  "introduction": "2-3 paragraphs setting the context, discussing recent performance, and stating the main investment thesis.",
  "technical_analysis": "3-4 paragraphs detailing the technical state. Discuss the meaning of the bias, trend, momentum, and volume scores. Explain the significance of the top signals and key support/resistance levels. Mention specific indicators like RSI, MACD, and Moving Averages.",
  "market_context": "2 paragraphs discussing the stock in its sector/industry context. How is the sector performing? Are there any broader market trends affecting this stock?",
  "investment_outlook": {{
    "short_term": "1-2 paragraphs on the short-term (days to weeks) outlook. What are the immediate catalysts or risks?",
    "long_term": "1-2 paragraphs on the long-term (months to years) outlook. What are the fundamental drivers or structural challenges?"
  }},
  "risk_factors": "A paragraph discussing the primary risks to the investment thesis (e.g., technical breakdown, market shift, competitive pressure).",
  "conclusion": "A final paragraph summarizing the analysis and providing a clear recommendation (e.g., Buy, Sell, Hold, Wait for Confirmation) with a concluding thought.",
  "tone_analysis": {{
    "sentiment": "Bullish",
    "confidence": "Medium",
    "timeframe": "Swing Trade (Weeks)"
  }}
}}
"""

            response = self.genai_client.models.generate_content(
                model='gemini-1.5-flash',  # Using a more advanced model for structured JSON
                contents=prompt,
                generation_config={
                    "response_mime_type": "application/json",
                }
            )
            
            # The response should be a valid JSON string now
            return json.loads(response.text)
            
        except Exception as e:
            return { "error": f"Analysis error: {str(e)}" }

    def _format_top_signals_for_article(self):
        """Helper to format top signals for the article prompt"""
        if not self.signals:
            return "No significant signals detected."
        
        formatted_signals = []
        for sig in self.signals[:5]: # Top 5 signals
            score = sig.get('ai_score', 'N/A')
            formatted_signals.append(f"- {sig['signal']} (Score: {score}): {sig['desc']}")
        return "\n".join(formatted_signals)
    
    def _get_trade_recommendations(self, current, analysis):
        """Get specific trade recommendations"""
        try:
            # Calculate option strikes
            current_price = current['Close']
            atr = current.get('ATR', current_price * 0.02)
            
            # Credit spreads (5-wide, 30+ DTE)
            put_spread_short = round(current_price - (2 * atr), 0)
            put_spread_long = put_spread_short - 5
            
            call_spread_short = round(current_price + (2 * atr), 0)
            call_spread_long = call_spread_short + 5
            
            prompt = f"""Generate specific trade recommendations for {self.symbol}.

MARKET DATA:
- Price: ${current_price:.2f}
- ATR: ${atr:.2f}
- Volatility: {current.get('Volatility', 30):.0f}%
- Overall Bias: {analysis['overall_bias']}
- Trend Strength: {analysis['indicator_ratings']['trend_strength']}/100

SUGGESTED SPREADS (5-wide, 30+ DTE):
Put Credit Spread: Sell ${put_spread_short:.0f} / Buy ${put_spread_long:.0f}
Call Credit Spread: Sell ${call_spread_short:.0f} / Buy ${call_spread_long:.0f}

Based on technicals, recommend:
1. BEST STRATEGY: Which spread or direction?
2. ENTRY CRITERIA: Specific conditions to enter
3. RISK MANAGEMENT: Stop loss / position sizing
4. PROBABILITY: Win probability estimate
5. ALTERNATIVE: Second-best strategy

Return JSON:
{
  "primary_strategy": {
    "type": "put_credit_spread" or "call_credit_spread",
    "short_strike": 0,
    "long_strike": 0,
    "rationale": "why this trade",
    "entry_criteria": "when to enter",
    "max_loss": "$X",
    "target_profit": "$X",
    "win_probability": "X%"
  },
  "alternative_strategy": {...},
  "risk_notes": "key risks"
}"""

            response = self.genai_client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            
            response_text = response.text.strip()
            
            # Try to parse JSON
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                response_text = response_text[start_idx:end_idx+1]
                
            try:
                return json.loads(response_text)
            except:
                return {'raw_text': response_text}
                
        except Exception as e:
            return {'error': str(e)}
    
    def upload_gemini_analysis_to_gcp(self, analysis):
        """Upload Gemini analysis to Google Cloud Storage"""
        if not analysis:
            return

        print("   ☁️ Uploading Gemini analysis to GCP...")
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.gcp_bucket)
            date_folder = datetime.now().strftime('%Y-%m-%d')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            gemini_data = {
                'symbol': self.symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'date': date_folder,
                'analysis': analysis.get('market_analysis', 'No analysis available'),
                'signal_count': len(self.signals),
                'signals_analyzed': [
                    {
                        'signal': s['signal'],
                        'desc': s['desc'],
                        'strength': s['strength'],
                        'category': s['category']
                    }
                    for s in self.signals[:50]  # Top 50 signals
                ]
            }

            if 'trade_recommendations' in analysis:
                trade_rec = analysis['trade_recommendations']
                if isinstance(trade_rec, dict) and 'primary_strategy' in trade_rec:
                    gemini_data['recommendation'] = json.dumps(trade_rec['primary_strategy'])

            blob = bucket.blob(f'daily/{date_folder}/{self.symbol}_gemini_analysis_{timestamp}.json')
            blob.upload_from_string(json.dumps(gemini_data, indent=2, default=str), content_type='application/json')
            print(f"     ✅ Uploaded: {self.symbol}_gemini_analysis_{timestamp}.json")

        except Exception as e:
            print(f"   ❌ GCP Upload Error for Gemini analysis: {str(e)}")

    def save_gemini_analysis_locally(self, analysis):
        """Save Gemini analysis to a local file"""
        if not analysis:
            return

        try:
            filename = self._generate_filename('gemini_analysis', 'json')
            filepath = os.path.join(self.date_folder, filename)
            
            with open(filepath, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            print(f"   💾 Saved Gemini analysis: {filename}")

        except Exception as e:
            print(f"   ❌ Error saving Gemini analysis locally: {str(e)}")
    
    def save_to_gcp(self):
        """Upload comprehensive data to GCP in Next.js dashboard compatible format"""
        print(f"\n☁️  Uploading to GCP: {self.gcp_bucket}...")
        
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.gcp_bucket)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            date_folder = datetime.now().strftime('%Y-%m-%d')
            current = self.data.iloc[-1]
            
            # 1. SIGNALS FILE (signals_SYMBOL_timestamp.json)
            # Format expected by Next.js dashboard
            signals_data = {
                'symbol': self.symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'date': date_folder,
                'price': float(current['Close']),
                'change_pct': float(current.get('Price_Change', 0)),
                'volume': int(current['Volume']),
                'indicators': {
                    'RSI': float(current['RSI']),
                    'MACD': float(current['MACD']),
                    'ADX': float(current.get('ADX', 0)),
                    'Stochastic': float(current['Stoch_K']),
                    'CCI': float(current.get('CCI', 0)),
                    'MFI': float(current.get('MFI', 50)),
                    'BB_Position': float(current.get('BB_Position', 0.5)),
                    'Volatility': float(current.get('Volatility', 20))
                },
                'moving_averages': {
                    'SMA_10': float(current['SMA_10']),
                    'SMA_20': float(current['SMA_20']),
                    'SMA_50': float(current['SMA_50']),
                    'SMA_200': float(current.get('SMA_200', 0)) if not pd.isna(current.get('SMA_200')) else None,
                    'EMA_10': float(current['EMA_10']),
                    'EMA_20': float(current['EMA_20'])
                },
                'signals': self.signals,
                'signal_count': len(self.signals),
                'bullish_count': sum(1 for s in self.signals if 'BULLISH' in s['strength']),
                'bearish_count': sum(1 for s in self.signals if 'BEARISH' in s['strength'])
            }
            
            blob = bucket.blob(f'daily/{date_folder}/signals_{self.symbol}_{timestamp}.json')
            blob.upload_from_string(json.dumps(signals_data, indent=2, default=str), content_type='application/json')
            print(f"  ✅ Uploaded: signals_{self.symbol}_{timestamp}.json")
            
            # 2. GEMINI ANALYSIS FILE (SYMBOL_gemini_analysis_timestamp.json)
            # Format expected by Next.js dashboard
            if hasattr(self, 'ai_analysis') and self.ai_analysis:
                gemini_data = {
                    'symbol': self.symbol,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'date': date_folder,
                    'analysis': self.ai_analysis.get('market_analysis', 'No analysis available'),
                    'signal_count': len(self.signals),
                    'signals_analyzed': [
                        {
                            'signal': s['signal'],
                            'desc': s['desc'],
                            'strength': s['strength'],
                            'category': s['category']
                        }
                        for s in self.signals[:50]  # Top 50 signals
                    ]
                }
                
                # Add trade recommendation if available
                if 'trade_recommendations' in self.ai_analysis:
                    trade_rec = self.ai_analysis['trade_recommendations']
                    if isinstance(trade_rec, dict) and 'primary_strategy' in trade_rec:
                        gemini_data['recommendation'] = json.dumps(trade_rec['primary_strategy'])
                
                blob = bucket.blob(f'daily/{date_folder}/{self.symbol}_gemini_analysis_{timestamp}.json')
                blob.upload_from_string(json.dumps(gemini_data, indent=2, default=str), content_type='application/json')
                print(f"  ✅ Uploaded: {self.symbol}_gemini_analysis_{timestamp}.json")
            
            # 3. COMPLETE ANALYSIS (for your own use)
            complete_analysis = self.generate_comprehensive_analysis()
            if hasattr(self, 'ai_analysis') and self.ai_analysis:
                complete_analysis['ai_analysis'] = self.ai_analysis
            
            blob = bucket.blob(f'daily/{date_folder}/{self.symbol}_complete_analysis_{timestamp}.json')
            blob.upload_from_string(json.dumps(complete_analysis, indent=2, default=str), content_type='application/json')
            print(f"  ✅ Uploaded: {self.symbol}_complete_analysis_{timestamp}.json")
            
            # 4. Technical data CSV
            csv_data = self.data.to_csv()
            blob = bucket.blob(f'daily/{date_folder}/{self.symbol}_technical_data_{timestamp}.csv')
            blob.upload_from_string(csv_data, content_type='text/csv')
            print(f"  ✅ Uploaded: {self.symbol}_technical_data_{timestamp}.csv")
            
            print(f"\n✅ GCP upload complete: gs://{self.gcp_bucket}/daily/{date_folder}/")
            print(f"   Dashboard will show: {self.symbol}")
            return True
            
        except Exception as e:
            print(f"❌ GCP Upload Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_locally(self):
        """Save comprehensive data locally"""
        print(f"\n💾 Saving files locally...")
        
        try:
            # Generate comprehensive analysis
            analysis = self.generate_comprehensive_analysis()
            
            # Add AI analysis if available
            if hasattr(self, 'ai_analysis') and self.ai_analysis:
                analysis['ai_analysis'] = self.ai_analysis
            
            # 1. Complete analysis JSON
            json_filename = self._generate_filename('complete_analysis', 'json')
            json_path = os.path.join(self.date_folder, json_filename)
            with open(json_path, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"  ✅ Saved: {json_filename}")
            
            # 2. Technical data CSV
            csv_filename = self._generate_filename('technical_data', 'csv')
            csv_path = os.path.join(self.date_folder, csv_filename)
            self.data.to_csv(csv_path)
            print(f"  ✅ Saved: {csv_filename}")
            
            # 3. Readable report
            report_filename = self._generate_filename('analysis_report', 'txt')
            report_path = os.path.join(self.date_folder, report_filename)
            with open(report_path, 'w') as f:
                f.write(self._generate_text_report(analysis))
            print(f"  ✅ Saved: {report_filename}")
            
            print(f"\n✅ Files saved to: {self.date_folder}")
            return True
            
        except Exception as e:
            print(f"❌ Local Save Error: {str(e)}")
            return False
    
    def _generate_text_report(self, analysis):
        """Generate readable text report"""
        report = f"""
{'='*80}
COMPREHENSIVE TECHNICAL ANALYSIS - {self.symbol}
{'='*80}

PRICE DATA:
  Current: ${analysis['price_data']['current_price']:.2f}
  Change: {analysis['price_data']['change']:.2f}%
  Volume: {analysis['price_data']['volume']:,}

OVERALL BIAS: {analysis['overall_bias']} (Score: {analysis['bias_score']}/100)

INDICATOR RATINGS:
  Trend Strength:      {analysis['indicator_ratings']['trend_strength']}/100
  Momentum:            {analysis['indicator_ratings']['momentum']}/100
  Volatility:          {analysis['indicator_ratings']['volatility']}/100
  Volume:              {analysis['indicator_ratings']['volume']}/100
  Support/Resistance:  {analysis['indicator_ratings']['support_resistance']}/100
  Oversold/Overbought: {analysis['indicator_ratings']['oversold_overbought']}/100

SIGNAL SUMMARY:
  Total Signals: {analysis['signal_summary']['total_signals']}
  Bullish: {analysis['signal_summary']['bullish_signals']}
  Bearish: {analysis['signal_summary']['bearish_signals']}
  Neutral: {analysis['signal_summary']['neutral_signals']}

TOP 20 RANKED SIGNALS:
{'='*80}
"""
        for i, sig in enumerate(self.signals[:20], 1):
            report += f"""
#{i} [{sig.get('ai_score', 'N/A')}/100] {sig['signal']}
   {sig['desc']}
   Category: {sig['category']} | Strength: {sig['strength']}
   AI: {sig.get('ai_reasoning', 'N/A')}
{'-'*80}
"""
        
        return report


def main():
    """Main execution function to run the advanced technical analysis."""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # --- Configuration ---
    SYMBOL = 'QQQ'
    PERIOD = '1y'
    UPLOAD_TO_GCP = True
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    # ---------------------

    print("=" * 80)
    print(f"🚀 ADVANCED TECHNICAL SCANNER - {SYMBOL}")
    print("   200+ Signals | AI Analysis | Options Strategies")
    print("=" * 80)
    
    try:
        analyzer = AdvancedTechnicalAnalyzer(
            symbol=SYMBOL,
            period=PERIOD,
            gemini_api_key=GEMINI_API_KEY
        )
        
        # Execute pipeline
        print("\n📊 Fetching market data...")
        analyzer.fetch_data()
        
        print("\n🔧 Calculating 50+ indicators...")
        analyzer.calculate_indicators()
        
        print("\n🎯 Detecting 200+ signals...")
        analyzer.detect_signals()
        
        print("\n🤖 Running AI analysis...")
        print(f"   [DEBUG] GEMINI_API_KEY found: {bool(GEMINI_API_KEY)}")
        if GEMINI_API_KEY:
            print("   [DEBUG] Calling analyzer.analyze_with_ai()...")
            analyzer.ai_analysis = analyzer.analyze_with_ai()
            print("   [DEBUG] AI analysis result:")
            print(json.dumps(analyzer.ai_analysis, indent=2, default=str))
        else:
            print("   ⚠️  No API key - skipping AI analysis")
        
        print("\n💾 Saving results...")
        print("   [DEBUG] Calling analyzer.save_locally()...")
        save_status = analyzer.save_locally()
        print(f"   [DEBUG] analyzer.save_locally() returned: {save_status}")
        
        if UPLOAD_TO_GCP:
            try:
                analyzer.save_to_gcp()
            except Exception as e:
                print(f"   ⚠️  GCP upload skipped: {str(e)[:50]}")
        
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 80)
        
        # Display summary
        analysis = analyzer.generate_comprehensive_analysis()
        print(f"\n📊 {SYMBOL} Summary:")
        print(f"   Price: ${analysis['price_data']['current_price']:.2f}")
        print(f"   Bias: {analysis['overall_bias']}")
        print(f"   Signals: {analysis['signal_summary']['total_signals']} total")
        if analyzer.signals:
            print(f"   Top Signal: {analyzer.signals[0]['signal']} [{analyzer.signals[0].get('ai_score', 'N/A')}]")

    except Exception as e:
        print(f"\n❌ An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()


# Example usage
if __name__ == "__main__":
    main()
