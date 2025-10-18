"""
Advanced Technical Analysis Scanner using yfinance
Monitors 20 indicators and detects 20 significant technical moments
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
        """Calculate 20 technical indicators"""
        df = self.data.copy()
        
        print("\n🔧 Calculating 20 Technical Indicators...")
        
        # 1. Simple Moving Averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # 2. Exponential Moving Averages
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # 3. RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 4. MACD (Moving Average Convergence Divergence)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # 5. Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # 6. Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # 7. ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # 8. ADX (Average Directional Index)
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr14 = true_range.rolling(14).sum()
        plus_di = 100 * (plus_dm.rolling(14).sum() / tr14)
        minus_di = 100 * (minus_dm.rolling(14).sum() / tr14)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(14).mean()
        
        # 9. CCI (Commodity Channel Index)
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        
        # 10. Williams %R
        df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
        
        # 11. OBV (On-Balance Volume)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # 12. Volume Moving Average
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # 13. VWAP (Volume Weighted Average Price)
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        # 14. Rate of Change (ROC)
        df['ROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100
        
        # 15. Money Flow Index (MFI)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        mfi_ratio = positive_flow / negative_flow
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        
        # 16. Parabolic SAR (simplified)
        df['SAR'] = df['Close'].shift(1)  # Simplified version
        
        # 17. Ichimoku Cloud components
        high_9 = df['High'].rolling(window=9).max()
        low_9 = df['Low'].rolling(window=9).min()
        df['Tenkan'] = (high_9 + low_9) / 2
        high_26 = df['High'].rolling(window=26).max()
        low_26 = df['Low'].rolling(window=26).min()
        df['Kijun'] = (high_26 + low_26) / 2
        
        # 18. Momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        
        # 19. Price Distance from 200 MA
        df['Distance_200MA'] = ((df['Close'] - df['SMA_200']) / df['SMA_200']) * 100
        
        # 20. Volatility (20-day standard deviation)
        df['Volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
        
        self.data = df
        print("✅ All 20 indicators calculated")
        
        return df
    
    def detect_signals(self):
        """Detect 20 significant technical moments/signals"""
        df = self.data.copy()
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        signals = []
        
        print("\n🎯 Detecting 20 Technical Signals...")
        print("=" * 80)
        
        # 1. Golden Cross (50 MA crosses above 200 MA)
        if len(df) > 200:
            if previous['SMA_50'] <= previous['SMA_200'] and current['SMA_50'] > current['SMA_200']:
                signals.append({
                    'signal': '🟢 GOLDEN CROSS',
                    'description': '50 MA crossed above 200 MA',
                    'strength': 'STRONG BULLISH',
                    'value': f"50MA: ${current['SMA_50']:.2f}, 200MA: ${current['SMA_200']:.2f}"
                })
        
        # 2. Death Cross (50 MA crosses below 200 MA)
        if len(df) > 200:
            if previous['SMA_50'] >= previous['SMA_200'] and current['SMA_50'] < current['SMA_200']:
                signals.append({
                    'signal': '🔴 DEATH CROSS',
                    'description': '50 MA crossed below 200 MA',
                    'strength': 'STRONG BEARISH',
                    'value': f"50MA: ${current['SMA_50']:.2f}, 200MA: ${current['SMA_200']:.2f}"
                })
        
        # 3. 10 EMA crosses 20 EMA (Bullish)
        if previous['EMA_10'] <= previous['EMA_20'] and current['EMA_10'] > current['EMA_20']:
            signals.append({
                'signal': '🟢 SHORT-TERM BULLISH CROSS',
                'description': '10 EMA crossed above 20 EMA',
                'strength': 'BULLISH',
                'value': f"10EMA: ${current['EMA_10']:.2f}, 20EMA: ${current['EMA_20']:.2f}"
            })
        
        # 4. 10 EMA crosses 20 EMA (Bearish)
        if previous['EMA_10'] >= previous['EMA_20'] and current['EMA_10'] < current['EMA_20']:
            signals.append({
                'signal': '🔴 SHORT-TERM BEARISH CROSS',
                'description': '10 EMA crossed below 20 EMA',
                'strength': 'BEARISH',
                'value': f"10EMA: ${current['EMA_10']:.2f}, 20EMA: ${current['EMA_20']:.2f}"
            })
        
        # 5. RSI Oversold (< 30)
        if current['RSI'] < 30:
            signals.append({
                'signal': '🟢 RSI OVERSOLD',
                'description': 'RSI below 30 - potential bounce',
                'strength': 'BULLISH',
                'value': f"RSI: {current['RSI']:.2f}"
            })
        
        # 6. RSI Overbought (> 70)
        if current['RSI'] > 70:
            signals.append({
                'signal': '🔴 RSI OVERBOUGHT',
                'description': 'RSI above 70 - potential pullback',
                'strength': 'BEARISH',
                'value': f"RSI: {current['RSI']:.2f}"
            })
        
        # 7. MACD Bullish Crossover
        if previous['MACD'] <= previous['MACD_Signal'] and current['MACD'] > current['MACD_Signal']:
            signals.append({
                'signal': '🟢 MACD BULLISH CROSSOVER',
                'description': 'MACD crossed above signal line',
                'strength': 'BULLISH',
                'value': f"MACD: {current['MACD']:.4f}, Signal: {current['MACD_Signal']:.4f}"
            })
        
        # 8. MACD Bearish Crossover
        if previous['MACD'] >= previous['MACD_Signal'] and current['MACD'] < current['MACD_Signal']:
            signals.append({
                'signal': '🔴 MACD BEARISH CROSSOVER',
                'description': 'MACD crossed below signal line',
                'strength': 'BEARISH',
                'value': f"MACD: {current['MACD']:.4f}, Signal: {current['MACD_Signal']:.4f}"
            })
        
        # 9. Bollinger Band Squeeze (narrow bands)
        bb_width_avg = df['BB_Width'].tail(50).mean()
        if current['BB_Width'] < bb_width_avg * 0.7:
            signals.append({
                'signal': '⚠️ BOLLINGER BAND SQUEEZE',
                'description': 'Bands are narrowing - expect breakout',
                'strength': 'NEUTRAL - BREAKOUT PENDING',
                'value': f"Width: ${current['BB_Width']:.2f}"
            })
        
        # 10. Price touches Lower Bollinger Band
        if current['Close'] <= current['BB_Lower'] * 1.01:
            signals.append({
                'signal': '🟢 TOUCHING LOWER BB',
                'description': 'Price at or below lower Bollinger Band',
                'strength': 'BULLISH',
                'value': f"Price: ${current['Close']:.2f}, Lower BB: ${current['BB_Lower']:.2f}"
            })
        
        # 11. Price touches Upper Bollinger Band
        if current['Close'] >= current['BB_Upper'] * 0.99:
            signals.append({
                'signal': '🔴 TOUCHING UPPER BB',
                'description': 'Price at or above upper Bollinger Band',
                'strength': 'BEARISH',
                'value': f"Price: ${current['Close']:.2f}, Upper BB: ${current['BB_Upper']:.2f}"
            })
        
        # 12. Stochastic Oversold
        if current['Stoch_K'] < 20:
            signals.append({
                'signal': '🟢 STOCHASTIC OVERSOLD',
                'description': 'Stochastic below 20',
                'strength': 'BULLISH',
                'value': f"Stoch K: {current['Stoch_K']:.2f}"
            })
        
        # 13. Stochastic Overbought
        if current['Stoch_K'] > 80:
            signals.append({
                'signal': '🔴 STOCHASTIC OVERBOUGHT',
                'description': 'Stochastic above 80',
                'strength': 'BEARISH',
                'value': f"Stoch K: {current['Stoch_K']:.2f}"
            })
        
        # 14. High Volume Spike
        if current['Volume'] > current['Volume_MA'] * 2:
            signals.append({
                'signal': '📊 HIGH VOLUME SPIKE',
                'description': 'Volume 2x above average',
                'strength': 'SIGNIFICANT ACTIVITY',
                'value': f"Volume: {current['Volume']:,.0f}, Avg: {current['Volume_MA']:,.0f}"
            })
        
        # 15. ADX Strong Trend
        if current['ADX'] > 25:
            signals.append({
                'signal': '💪 STRONG TREND',
                'description': f"ADX above 25 - strong {'up' if current['Close'] > current['SMA_50'] else 'down'}trend",
                'strength': 'TRENDING',
                'value': f"ADX: {current['ADX']:.2f}"
            })
        
        # 16. CCI Extreme Oversold
        if current['CCI'] < -200:
            signals.append({
                'signal': '🟢 CCI EXTREME OVERSOLD',
                'description': 'CCI below -200',
                'strength': 'STRONG BULLISH',
                'value': f"CCI: {current['CCI']:.2f}"
            })
        
        # 17. CCI Extreme Overbought
        if current['CCI'] > 200:
            signals.append({
                'signal': '🔴 CCI EXTREME OVERBOUGHT',
                'description': 'CCI above 200',
                'strength': 'STRONG BEARISH',
                'value': f"CCI: {current['CCI']:.2f}"
            })
        
        # 18. MFI Oversold (Money Flow)
        if current['MFI'] < 20:
            signals.append({
                'signal': '🟢 MONEY FLOW OVERSOLD',
                'description': 'MFI below 20 - buying pressure low',
                'strength': 'BULLISH',
                'value': f"MFI: {current['MFI']:.2f}"
            })
        
        # 19. Williams %R Extreme
        if current['Williams_R'] < -80:
            signals.append({
                'signal': '🟢 WILLIAMS %R OVERSOLD',
                'description': 'Williams %R below -80',
                'strength': 'BULLISH',
                'value': f"Williams %R: {current['Williams_R']:.2f}"
            })
        
        # 20. High Volatility Warning
        if current['Volatility'] > 50:
            signals.append({
                'signal': '⚠️ HIGH VOLATILITY',
                'description': 'Annualized volatility above 50%',
                'strength': 'CAUTION',
                'value': f"Volatility: {current['Volatility']:.2f}%"
            })
        
        self.signals = signals
        
        # Display all detected signals
        if signals:
            print(f"\n✅ Detected {len(signals)} Active Signals:\n")
            for i, sig in enumerate(signals, 1):
                print(f"{i}. {sig['signal']}")
                print(f"   📝 {sig['description']}")
                print(f"   💪 Strength: {sig['strength']}")
                print(f"   📊 {sig['value']}")
                print()
        else:
            print("❌ No significant signals detected at this time")
        
        return signals
    
    def generate_summary(self):
        """Generate comprehensive analysis summary"""
        current = self.data.iloc[-1]
        
        print("\n" + "=" * 80)
        print(f"📊 TECHNICAL ANALYSIS SUMMARY FOR {self.symbol}")
        print("=" * 80)
        
        print(f"\n💰 Current Price: ${current['Close']:.2f}")
        print(f"📅 Analysis Date: {current.name.strftime('%Y-%m-%d')}")
        
        print("\n📈 Moving Averages:")
        print(f"   10 EMA: ${current['EMA_10']:.2f}")
        print(f"   20 EMA: ${current['EMA_20']:.2f}")
        print(f"   50 SMA: ${current['SMA_50']:.2f}")
        if not pd.isna(current['SMA_200']):
            print(f"  200 SMA: ${current['SMA_200']:.2f}")
        
        print("\n📊 Oscillators:")
        print(f"   RSI (14): {current['RSI']:.2f}")
        print(f"   Stochastic: {current['Stoch_K']:.2f}")
        print(f"   CCI: {current['CCI']:.2f}")
        print(f"   Williams %R: {current['Williams_R']:.2f}")
        print(f"   MFI: {current['MFI']:.2f}")
        
        print("\n📉 Volatility & Momentum:")
        print(f"   ATR: ${current['ATR']:.2f}")
        print(f"   ADX: {current['ADX']:.2f}")
        print(f"   Volatility: {current['Volatility']:.2f}%")
        print(f"   ROC: {current['ROC']:.2f}%")
        
        print("\n🎯 Bollinger Bands:")
        print(f"   Upper: ${current['BB_Upper']:.2f}")
        print(f"   Middle: ${current['BB_Middle']:.2f}")
        print(f"   Lower: ${current['BB_Lower']:.2f}")
        print(f"   Position: {current['BB_Position']:.2%}")
        
        # Overall trend assessment
        print("\n🔍 Trend Assessment:")
        bullish_count = sum(1 for s in self.signals if 'BULLISH' in s['strength'])
        bearish_count = sum(1 for s in self.signals if 'BEARISH' in s['strength'])
        
        if bullish_count > bearish_count:
            print("   Overall: 🟢 BULLISH BIAS")
        elif bearish_count > bullish_count:
            print("   Overall: 🔴 BEARISH BIAS")
        else:
            print("   Overall: 🟡 NEUTRAL/MIXED")
        
        print(f"   Bullish Signals: {bullish_count}")
        print(f"   Bearish Signals: {bearish_count}")
        
        print("\n" + "=" * 80)

def main():
    """Main execution function"""
    
    # Configuration
    SYMBOL = 'RGTI'  # Change to any stock symbol
    PERIOD = '1y'     # 1 year of data
    
    print("=" * 80)
    print("🚀 ADVANCED TECHNICAL ANALYSIS SCANNER")
    print("=" * 80)
    
    try:
        # Initialize analyzer
        analyzer = TechnicalAnalyzer(SYMBOL, PERIOD)
        
        # Fetch data
        analyzer.fetch_data()
        
        # Calculate indicators
        analyzer.calculate_indicators()
        
        # Detect signals
        analyzer.detect_signals()
        
        # Generate summary
        analyzer.generate_summary()
        
        print("\n✅ Analysis Complete!")
        
        # Optional: Save data to CSV
        analyzer.data.to_csv(f'{SYMBOL}_technical_analysis.csv')
        # print(f"\n💾 Data saved to {SYMBOL}_technical_analysis.csv")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()