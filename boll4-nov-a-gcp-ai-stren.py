# good for signal strength 

"""
Advanced Technical Analysis Scanner with AI-Powered Signal Ranking (1-100)
Gemini AI scores each signal based on its own judgment and market context
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
    
    def rank_signals_with_ai(self):
        """Use Gemini AI to score each signal from 1-100 based on its judgment"""
        if not self.genai_client:
            print("\n⚠️  Gemini API key not provided. Skipping AI ranking.")
            # Add default scores if no AI
            for signal in self.signals:
                signal['ai_score'] = 50
                signal['ai_reasoning'] = 'No AI scoring available'
            return
        
        print("\n🤖 AI is scoring all signals (1-100)...")
        
        try:
            current = self.data.iloc[-1]
            
            # Create comprehensive prompt for AI scoring
            prompt = f"""You are an expert technical analyst scoring trading signals for {self.symbol}.

CURRENT MARKET DATA:
- Price: ${current['Close']:.2f}
- Daily Change: {current['Price_Change']:.2f}%
- RSI: {current['RSI']:.1f}
- MACD: {current['MACD']:.4f}
- ADX: {current['ADX']:.1f}
- Volatility: {current['Volatility']:.1f}%
- Volume vs 20-day avg: {(current['Volume'] / current['Volume_MA_20'] * 100):.0f}%

SIGNALS TO SCORE:
"""
            for i, sig in enumerate(self.signals, 1):
                prompt += f"\n{i}. {sig['signal']} - {sig['desc']} ({sig['strength']}) [{sig['category']}]"
            
            prompt += """

TASK: Score each signal from 1-100 based on:
- Actionability (how tradeable is this signal?)
- Reliability (historical success rate of this pattern)
- Timing (is this the right moment?)
- Risk/Reward (potential upside vs downside)
- Context (does it align with overall market conditions?)

RESPONSE FORMAT (JSON):
{
  "scores": [
    {"signal_number": 1, "score": 85, "reasoning": "Strong reversal signal with high volume confirmation"},
    {"signal_number": 2, "score": 72, "reasoning": "Moderate bullish signal but RSI shows overbought"},
    ...
  ],
  "top_signal": {
    "signal_number": 1,
    "why": "Highest probability setup with best risk/reward"
  }
}

Score EVERY signal. Be critical - don't give everything high scores. Use the full 1-100 range."""

            # Call Gemini API
            response = self.genai_client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Extract JSON from markdown code blocks if present
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            scores_data = json.loads(response_text)
            
            # Apply scores to signals
            for score_item in scores_data['scores']:
                sig_num = score_item['signal_number'] - 1  # 0-indexed
                if 0 <= sig_num < len(self.signals):
                    self.signals[sig_num]['ai_score'] = score_item['score']
                    self.signals[sig_num]['ai_reasoning'] = score_item['reasoning']
            
            # Sort signals by AI score
            self.signals.sort(key=lambda x: x.get('ai_score', 0), reverse=True)
            
            # Add rank
            for rank, signal in enumerate(self.signals, 1):
                signal['rank'] = rank
            
            # Store top signal info
            self.top_signal_info = scores_data.get('top_signal', {})
            
            print(f"✅ AI scored {len(self.signals)} signals")
            print(f"🏆 Top Signal: #{self.top_signal_info.get('signal_number', 'N/A')} - {self.top_signal_info.get('why', 'N/A')}")
            
        except Exception as e:
            print(f"❌ AI Scoring Error: {str(e)}")
            print(f"Response text: {response.text if 'response' in locals() else 'No response'}")
            # Fallback to default scores
            for signal in self.signals:
                signal['ai_score'] = 50
                signal['ai_reasoning'] = 'AI scoring failed'
    
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
        
        print("\n🎯 Scanning for Technical Alerts...")
        
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
        
        self.signals = signals
        
        print(f"✅ Detected {len(signals)} Active Signals")
        return signals
    
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
            
            # 2. Save signals JSON with AI scores
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
                'bearish_count': sum(1 for s in self.signals if 'BEARISH' in s['strength']),
                'top_signal_info': getattr(self, 'top_signal_info', {})
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
            
            # 4. Save ranked signals report
            ranked_filename = self._generate_filename('ranked_signals', 'txt')
            ranked_path = os.path.join(self.date_folder, ranked_filename)
            with open(ranked_path, 'w') as f:
                f.write(self.generate_ranked_signals_report())
            print(f"✅ Saved: {ranked_filename}")
            
            print(f"\n✅ All files saved to: {self.date_folder}")
            return True
            
        except Exception as e:
            print(f"❌ Local Save Error: {str(e)}")
            return False
    
    def generate_ranked_signals_report(self):
        """Generate a formatted report of ranked signals"""
        report = f"""
{'='*80}
RANKED SIGNALS REPORT - {self.symbol}
{'='*80}

AI-Scored Signals (1-100 Scale)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        if hasattr(self, 'top_signal_info') and self.top_signal_info:
            report += f"""🏆 TOP SIGNAL IDENTIFIED BY AI:
Signal #{self.top_signal_info.get('signal_number', 'N/A')}
Reason: {self.top_signal_info.get('why', 'N/A')}

"""
        
        report += f"{'='*80}\nALL SIGNALS (Ranked by AI Score):\n{'='*80}\n\n"
        
        for signal in self.signals:
            score = signal.get('ai_score', 'N/A')
            rank = signal.get('rank', '?')
            reasoning = signal.get('ai_reasoning', 'No reasoning provided')
            
            # Score indicator
            if isinstance(score, (int, float)):
                if score >= 80:
                    indicator = "🔥"
                elif score >= 60:
                    indicator = "⚡"
                elif score >= 40:
                    indicator = "📊"
                else:
                    indicator = "⚠️"
            else:
                indicator = "❓"
            
            report += f"""#{rank} {indicator} SCORE: {score}/100
Signal: {signal['signal']}
Description: {signal['desc']}
Category: {signal['category']} | Strength: {signal['strength']}
AI Analysis: {reasoning}
{'-'*80}

"""
        
        return report
    
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
        
        # Show top 10 AI-ranked signals
        top_signals = self.signals[:10] if len(self.signals) > 10 else self.signals
        for i, sig in enumerate(top_signals, 1):
            score = sig.get('ai_score', 'N/A')
            summary += f"\n{i}. [{score}/100] {sig['signal']}\n   {sig['desc']} - {sig['strength']}\n"
        
        if len(self.signals) > 10:
            summary += f"\n... and {len(self.signals) - 10} more signals\n"
        
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
        """Print summary to console with top AI-ranked signals"""
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
        
        # Show top 5 AI-ranked signals
        if self.signals:
            print(f"\n🏆 TOP 5 AI-RANKED SIGNALS:")
            print("=" * 80)
            for i, sig in enumerate(self.signals[:5], 1):
                score = sig.get('ai_score', 'N/A')
                print(f"{i}. [{score}/100] {sig['signal']}")
                print(f"   {sig['desc']}")
                print(f"   AI: {sig.get('ai_reasoning', 'No reasoning')[:60]}...")
                print()
        
        print("=" * 80)

def main():
    """Main execution with AI-powered signal ranking"""
    
    # Configuration
    SYMBOL = 'ORCL'
    PERIOD = '1y'
    GCP_BUCKET = 'ttb-bucket1'
    
    # Set your Gemini API key here or via environment variable
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    print("=" * 80)
    print("🚀 TECHNICAL SCANNER with AI SIGNAL RANKING (1-100)")
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
        
        # Step 4: AI ranks all signals (1-100)
        if GEMINI_API_KEY:
            analyzer.rank_signals_with_ai()
        else:
            print("\n⚠️  Set GEMINI_API_KEY for AI signal ranking")
        
        # Step 5: Print summary with top signals
        analyzer.print_summary()
        
        # Step 6: Save files locally (includes ranked report)
        save_success = analyzer.save_locally()
        
        # Step 7: Full AI analysis (optional)
        if GEMINI_API_KEY:
            print("\n" + "="*80)
            print("🤖 GETTING COMPREHENSIVE AI ANALYSIS")
            print("="*80)
            gemini_analysis = analyzer.analyze_with_gemini()
            
            if gemini_analysis:
                analyzer.save_gemini_analysis_locally(gemini_analysis)
        
        # Step 8: Upload to GCP (optional)
        try:
            upload_success = analyzer.upload_to_gcp()
        except Exception as e:
            print(f"\n⚠️  GCP upload skipped: {str(e)}")
        
        print("\n✅ Analysis Complete!")
        print(f"📂 Files saved in: {analyzer.date_folder}")
        print(f"\n📄 Files created:")
        print(f"   - Technical data CSV (all indicators)")
        print(f"   - Signals JSON (with AI scores)")
        print(f"   - Summary report TXT")
        print(f"   - Ranked signals report TXT (NEW!)")
        if GEMINI_API_KEY:
            print(f"   - Gemini AI analysis JSON & TXT")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def analyze_with_gemini(self):
    """Use Gemini AI to provide comprehensive market analysis"""
    if not self.genai_client:
        print("\n⚠️  Gemini API key not provided. Skipping AI analysis.")
        return None
    
    print("\n🤖 Getting comprehensive AI analysis...")
    
    try:
        current = self.data.iloc[-1]
        
        # Get top 5 signals for context
        top_signals = self.signals[:5] if len(self.signals) >= 5 else self.signals
        
        prompt = f"""You are an expert technical analyst providing a comprehensive trading analysis for {self.symbol}.

MARKET DATA:
- Price: ${current['Close']:.2f}
- Daily Change: {current['Price_Change']:.2f}%
- Volume vs avg: {(current['Volume'] / current['Volume_MA_20'] * 100):.0f}%

INDICATORS:
- RSI: {current['RSI']:.1f}
- MACD: {current['MACD']:.4f}
- ADX: {current['ADX']:.1f} (trend strength)
- Stochastic: {current['Stoch_K']:.1f}
- MFI: {current['MFI']:.1f}
- Volatility: {current['Volatility']:.1f}%

TOP 5 SIGNALS (AI-Ranked):
"""
        for i, sig in enumerate(top_signals, 1):
            score = sig.get('ai_score', 'N/A')
            prompt += f"\n{i}. [{score}/100] {sig['signal']} - {sig['desc']}"
        
        prompt += f"""

Total signals detected: {len(self.signals)}
Bullish signals: {sum(1 for s in self.signals if 'BULLISH' in s['strength'])}
Bearish signals: {sum(1 for s in self.signals if 'BEARISH' in s['strength'])}

PROVIDE:

1. MARKET BIAS (Bullish/Bearish/Neutral with confidence %)

2. TOP TRADE SETUP
   - Entry price
   - Stop loss
   - Target(s)
   - Risk/Reward ratio

3. KEY LEVELS TO WATCH
   - Support levels
   - Resistance levels
   - Breakout/breakdown zones

4. RISK FACTORS
   - What could invalidate this setup?
   - Key risks to monitor

5. TIMEFRAME RECOMMENDATION
   - Best timeframe for this setup
   - Expected hold time

6. ACTION PLAN
   - Should traders buy, sell, or wait?
   - Specific conditions to watch for

Be specific, actionable, and reference the actual indicator values and top signals."""

        response = self.genai_client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt
        )
        
        print("\n" + "="*80)
        print("🤖 COMPREHENSIVE AI ANALYSIS")
        print("="*80)
        print(response.text)
        print("="*80)
        
        return response.text
        
    except Exception as e:
        print(f"❌ Gemini Analysis Error: {str(e)}")
        return None

def save_gemini_analysis_locally(self, analysis):
    """Save Gemini analysis to local folder"""
    if not analysis:
        return
    
    try:
        analysis_data = {
            'symbol': self.symbol,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'analysis': analysis,
            'signal_count': len(self.signals),
            'top_signals': self.signals[:10],
            'top_signal_info': getattr(self, 'top_signal_info', {})
        }
        
        json_filename = self._generate_filename('ai_analysis', 'json')
        json_path = os.path.join(self.date_folder, json_filename)
        with open(json_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        print(f"✅ Saved: {json_filename}")
        
        txt_filename = self._generate_filename('ai_analysis', 'txt')
        txt_path = os.path.join(self.date_folder, txt_filename)
        with open(txt_path, 'w') as f:
            f.write(f"COMPREHENSIVE AI ANALYSIS - {self.symbol}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(analysis)
        print(f"✅ Saved: {txt_filename}")
        
    except Exception as e:
        print(f"❌ Error saving AI analysis: {str(e)}")

def upload_to_gcp(self):
    """Upload data and alerts to GCP bucket"""
    print(f"\n☁️  Uploading to GCP: {self.gcp_bucket}/daily...")
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.gcp_bucket)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        date_folder = datetime.now().strftime('%Y-%m-%d')
        
        # 1. Upload technical data CSV
        csv_data = self.data.to_csv()
        blob = bucket.blob(f'daily/{date_folder}/{self.symbol}_technical_data_{timestamp}.csv')
        blob.upload_from_string(csv_data, content_type='text/csv')
        print(f"✅ Uploaded: technical_data CSV")
        
        # 2. Upload signals JSON with AI scores
        current = self.data.iloc[-1]
        signals_data = {
            'symbol': self.symbol,
            'timestamp': timestamp,
            'date': date_folder,
            'price': float(current['Close']),
            'change_pct': float(current['Price_Change']),
            'volume': int(current['Volume']),
            'signals': self.signals,
            'signal_count': len(self.signals),
            'top_signal_info': getattr(self, 'top_signal_info', {})
        }
        
        blob = bucket.blob(f'daily/{date_folder}/{self.symbol}_signals_ranked_{timestamp}.json')
        blob.upload_from_string(json.dumps(signals_data, indent=2), content_type='application/json')
        print(f"✅ Uploaded: signals with AI rankings")
        
        # 3. Upload ranked report
        ranked_report = self.generate_ranked_signals_report()
        blob = bucket.blob(f'daily/{date_folder}/{self.symbol}_ranked_{timestamp}.txt')
        blob.upload_from_string(ranked_report, content_type='text/plain')
        print(f"✅ Uploaded: ranked signals report")
        
        print(f"\n✅ All files uploaded to gs://{self.gcp_bucket}/daily/{date_folder}/")
        return True
        
    except Exception as e:
        print(f"❌ GCP Upload Error: {str(e)}")
        return False

# Add these methods to TechnicalAnalyzer class
TechnicalAnalyzer.analyze_with_gemini = analyze_with_gemini
TechnicalAnalyzer.save_gemini_analysis_locally = save_gemini_analysis_locally
TechnicalAnalyzer.upload_to_gcp = upload_to_gcp

if __name__ == "__main__":
    main()


"""
===================================================================================
AI-POWERED SIGNAL RANKING SYSTEM (1-100)
===================================================================================

NEW FEATURES:
✅ Gemini AI scores EVERY signal from 1-100
✅ AI provides reasoning for each score
✅ Signals automatically ranked by AI score
✅ Top signal identification with explanation
✅ Comprehensive ranked signals report

HOW IT WORKS:
1. Scanner detects 100+ technical signals as before
2. AI analyzes ALL signals with market context
3. AI scores each signal 1-100 based on:
   - Actionability (can you trade this?)
   - Reliability (historical success rate)
   - Timing (is this the right moment?)
   - Risk/Reward (upside vs downside)
   - Market context (does it fit current conditions?)
4. Signals sorted by AI score (highest first)
5. AI identifies single best signal to act on

OUTPUT FILES:
📄 YYYY-MM-DD-SYMBOL-technical_data-HHMMSS.csv (all indicators)
📄 YYYY-MM-DD-SYMBOL-signals-HHMMSS.json (with AI scores)
📄 YYYY-MM-DD-SYMBOL-summary-HHMMSS.txt (overview)
📄 YYYY-MM-DD-SYMBOL-ranked_signals-HHMMSS.txt (AI rankings! NEW!)
📄 YYYY-MM-DD-SYMBOL-ai_analysis-HHMMSS.json (comprehensive analysis)
📄 YYYY-MM-DD-SYMBOL-ai_analysis-HHMMSS.txt (readable analysis)

SIGNAL JSON STRUCTURE:
{
  "signal": "MACD BULL CROSS",
  "desc": "MACD crossed above signal",
  "strength": "BULLISH",
  "category": "MACD",
  "ai_score": 87,                    <-- NEW!
  "ai_reasoning": "Strong timing...", <-- NEW!
  "rank": 1                           <-- NEW!
}

EXAMPLE RANKED OUTPUT:
===================================================================================
#1 🔥 SCORE: 92/100
Signal: MACD BULL CROSS
Description: MACD crossed above signal line
Category: MACD | Strength: BULLISH
AI Analysis: Strong reversal signal with volume confirmation. RSI recovering from
oversold provides excellent risk/reward setup.
-----------------------------------------------------------------------------------

#2 ⚡ SCORE: 85/100
Signal: RSI OVERSOLD
Description: RSI at 28.3
Category: RSI | Strength: BULLISH
AI Analysis: Oversold reading suggests bounce potential, but wait for momentum
confirmation before entering.
-----------------------------------------------------------------------------------

SCORE INDICATORS:
🔥 80-100: Highly actionable signals
⚡60-79:  Good signals, watch for confirmation
📊 40-59:  Moderate signals, neutral impact
⚠️  1-39:  Weak signals, low priority

AI DECISION MAKING:
- AI doesn't just score blindly - it considers:
  ✓ Signal confluence (multiple signals agreeing)
  ✓ Current market phase (trending vs ranging)
  ✓ Risk management (where to place stops)
  ✓ Recent market behavior
  ✓ Contradicting signals

USAGE:
1. Set GEMINI_API_KEY environment variable
2. Run: python scanner.py
3. Check the ranked_signals.txt file for prioritized alerts
4. AI identifies the single best setup to trade

SETUP:
pip install yfinance pandas numpy google-cloud-storage google-genai
export GEMINI_API_KEY="your-api-key"
python scanner.py

The AI makes the final call on which signals matter most!
===================================================================================
"""