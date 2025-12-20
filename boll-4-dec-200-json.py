"""
YFinance Signal Detector with Safe JSON/MD Export
Fetches data, calculates indicators, detects signals, exports to JSON and Markdown
Handles all edge cases for safe JSON serialization
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path


class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles pandas/numpy types safely"""
    
    def default(self, obj):
        # Handle pandas/numpy types
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            # Handle NaN, Inf, -Inf
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        
        # Try to convert to string as last resort
        try:
            return str(obj)
        except:
            return None


class SignalDetectorExporter:
    """
    Complete signal detection and export pipeline
    """
    
    def __init__(self, symbol, period='1y', output_dir='signal_reports'):
        self.symbol = symbol
        self.period = period
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.data = None
        self.signals = []
        self.current = None
        
    def fetch_data(self):
        """Fetch stock data from yfinance"""
        print(f"📊 Fetching {self.symbol} data...")
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(period=self.period)
        
        if self.data.empty:
            raise ValueError(f"No data found for {self.symbol}")
        
        print(f"✅ Fetched {len(self.data)} days")
        return self.data
    
    def calculate_indicators(self):
        """Calculate all technical indicators"""
        print("🔧 Calculating indicators...")
        df = self.data.copy()
        
        # Moving Averages
        for period in [10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # RSI
        for period in [9, 14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI_14']
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        for period in [10, 20, 30]:
            bb_middle = df['Close'].rolling(window=period).mean()
            bb_std = df['Close'].rolling(window=period).std()
            df[f'BB_{period}_Upper'] = bb_middle + (bb_std * 2)
            df[f'BB_{period}_Lower'] = bb_middle - (bb_std * 2)
            df[f'BB_{period}_Position'] = (df['Close'] - df[f'BB_{period}_Lower']) / (df[f'BB_{period}_Upper'] - df[f'BB_{period}_Lower'])
        
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
        
        # Volume
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change() * 100
        df['Price_Change_5d'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
        
        # 52-week levels
        df['High_52w'] = df['High'].rolling(window=252).max()
        df['Low_52w'] = df['Low'].rolling(window=252).min()
        
        # Distance from MAs
        for period in [20, 50, 200]:
            df[f'Dist_SMA_{period}'] = ((df['Close'] - df[f'SMA_{period}']) / df[f'SMA_{period}']) * 100
        
        self.data = df
        print("✅ Indicators calculated")
        return df
    
    def detect_signals(self):
        """Detect all technical signals with safe value extraction"""
        print("🎯 Detecting signals...")
        
        df = self.data.copy()
        current = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else current
        
        self.current = current  # Save for later use
        signals = []
        
        def safe_float(value):
            """Safely convert to float, handling NaN/Inf"""
            try:
                if pd.isna(value) or np.isinf(value):
                    return None
                return float(value)
            except:
                return None
        
        def safe_value(value):
            """Get safe value for JSON"""
            result = safe_float(value)
            return result if result is not None else 0.0
        
        # MA Crosses
        if len(df) > 200:
            if not pd.isna(current['SMA_50']) and not pd.isna(current['SMA_200']):
                if prev['SMA_50'] <= prev['SMA_200'] and current['SMA_50'] > current['SMA_200']:
                    signals.append({
                        'signal': 'GOLDEN CROSS',
                        'description': '50 MA crossed above 200 MA',
                        'strength': 'STRONG BULLISH',
                        'category': 'MA_CROSS',
                        'value': safe_value(current['SMA_50'])
                    })
        
        for fast, slow in [(10, 20), (20, 50)]:
            col_fast, col_slow = f'SMA_{fast}', f'SMA_{slow}'
            if col_fast in df.columns and col_slow in df.columns:
                if not pd.isna(current[col_fast]) and not pd.isna(current[col_slow]):
                    if prev[col_fast] <= prev[col_slow] and current[col_fast] > current[col_slow]:
                        signals.append({
                            'signal': f'{fast}/{slow} MA BULL CROSS',
                            'description': f'{fast} MA crossed above {slow} MA',
                            'strength': 'BULLISH',
                            'category': 'MA_CROSS',
                            'value': safe_value(current[col_fast])
                        })
        
        # RSI Signals
        for period in [9, 14, 21]:
            rsi_col = f'RSI_{period}' if f'RSI_{period}' in df.columns else 'RSI'
            if rsi_col in df.columns and not pd.isna(current[rsi_col]):
                rsi = safe_value(current[rsi_col])
                if rsi < 30:
                    signals.append({
                        'signal': f'RSI{period} OVERSOLD',
                        'description': f'RSI({period}): {rsi:.1f}',
                        'strength': 'BULLISH',
                        'category': 'RSI',
                        'value': rsi
                    })
                elif rsi > 70:
                    signals.append({
                        'signal': f'RSI{period} OVERBOUGHT',
                        'description': f'RSI({period}): {rsi:.1f}',
                        'strength': 'BEARISH',
                        'category': 'RSI',
                        'value': rsi
                    })
        
        # MACD Signals
        if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
            if not pd.isna(current['MACD']) and not pd.isna(current['MACD_Signal']):
                if prev['MACD'] <= prev['MACD_Signal'] and current['MACD'] > current['MACD_Signal']:
                    signals.append({
                        'signal': 'MACD BULL CROSS',
                        'description': 'MACD crossed above signal line',
                        'strength': 'STRONG BULLISH',
                        'category': 'MACD',
                        'value': safe_value(current['MACD'])
                    })
        
        # Bollinger Bands
        if 'BB_20_Upper' in df.columns and 'BB_20_Lower' in df.columns:
            if not pd.isna(current['BB_20_Upper']) and not pd.isna(current['BB_20_Lower']):
                if current['Close'] > current['BB_20_Upper']:
                    signals.append({
                        'signal': 'ABOVE UPPER BB',
                        'description': 'Price broke above upper Bollinger Band',
                        'strength': 'EXTREME BULLISH',
                        'category': 'BB_BREAKOUT',
                        'value': safe_value(current['Close'] - current['BB_20_Upper'])
                    })
                elif current['Close'] < current['BB_20_Lower']:
                    signals.append({
                        'signal': 'BELOW LOWER BB',
                        'description': 'Price broke below lower Bollinger Band',
                        'strength': 'EXTREME BEARISH',
                        'category': 'BB_BREAKOUT',
                        'value': safe_value(current['BB_20_Lower'] - current['Close'])
                    })
        
        # Volume Signals
        if 'Volume_MA_20' in df.columns and not pd.isna(current['Volume_MA_20']):
            vol_ratio = safe_value(current['Volume'] / current['Volume_MA_20'])
            if vol_ratio > 2:
                signals.append({
                    'signal': 'VOLUME SPIKE',
                    'description': f'Volume: {vol_ratio:.1f}x average',
                    'strength': 'SIGNIFICANT',
                    'category': 'VOLUME',
                    'value': vol_ratio
                })
        
        # Stochastic
        if 'Stoch_K' in df.columns and not pd.isna(current['Stoch_K']):
            stoch_k = safe_value(current['Stoch_K'])
            if stoch_k < 20:
                signals.append({
                    'signal': 'STOCHASTIC OVERSOLD',
                    'description': f'%K: {stoch_k:.1f}',
                    'strength': 'BULLISH',
                    'category': 'STOCHASTIC',
                    'value': stoch_k
                })
            elif stoch_k > 80:
                signals.append({
                    'signal': 'STOCHASTIC OVERBOUGHT',
                    'description': f'%K: {stoch_k:.1f}',
                    'strength': 'BEARISH',
                    'category': 'STOCHASTIC',
                    'value': stoch_k
                })
        
        # ADX Trend
        if 'ADX' in df.columns and not pd.isna(current['ADX']):
            adx = safe_value(current['ADX'])
            if adx > 25:
                direction = 'UP' if current['Plus_DI'] > current['Minus_DI'] else 'DOWN'
                signals.append({
                    'signal': f'STRONG {direction}TREND',
                    'description': f'ADX: {adx:.1f}',
                    'strength': 'TRENDING',
                    'category': 'ADX',
                    'value': adx
                })
        
        # Price Action
        if 'Price_Change' in df.columns and not pd.isna(current['Price_Change']):
            pc = safe_value(current['Price_Change'])
            if pc > 5:
                signals.append({
                    'signal': 'LARGE GAIN',
                    'description': f'+{pc:.1f}% today',
                    'strength': 'STRONG BULLISH',
                    'category': 'PRICE_ACTION',
                    'value': pc
                })
            elif pc < -5:
                signals.append({
                    'signal': 'LARGE LOSS',
                    'description': f'{pc:.1f}% today',
                    'strength': 'STRONG BEARISH',
                    'category': 'PRICE_ACTION',
                    'value': pc
                })
        
        # 52-week levels
        if all(col in df.columns for col in ['High_52w', 'Low_52w']):
            if not pd.isna(current['High_52w']) and not pd.isna(current['Low_52w']):
                if current['Close'] >= current['High_52w'] * 0.999:
                    signals.append({
                        'signal': '52-WEEK HIGH',
                        'description': f'At ${safe_value(current["Close"]):.2f}',
                        'strength': 'EXTREME BULLISH',
                        'category': 'RANGE',
                        'value': safe_value(current['Close'])
                    })
        
        # Distance from MAs
        for period in [20, 50, 200]:
            dist_col = f'Dist_SMA_{period}'
            if dist_col in df.columns and not pd.isna(current[dist_col]):
                dist = safe_value(current[dist_col])
                if dist > 10:
                    signals.append({
                        'signal': f'OVEREXTENDED ABOVE {period}MA',
                        'description': f'{dist:.1f}% above {period}MA',
                        'strength': 'BEARISH',
                        'category': 'MA_DISTANCE',
                        'value': dist
                    })
                elif dist < -10:
                    signals.append({
                        'signal': f'OVEREXTENDED BELOW {period}MA',
                        'description': f'{abs(dist):.1f}% below {period}MA',
                        'strength': 'BULLISH',
                        'category': 'MA_DISTANCE',
                        'value': dist
                    })
        
        self.signals = signals
        print(f"✅ Detected {len(signals)} signals")
        return signals
    
    def export_json(self):
        """Export signals and data to JSON with safe serialization"""
        print("💾 Exporting to JSON...")
        
        def safe_float(val):
            """Safe float conversion"""
            try:
                if pd.isna(val) or np.isinf(val):
                    return None
                return float(val)
            except:
                return None
        
        current = self.current
        
        # Build safe JSON structure
        data = {
            'metadata': {
                'symbol': self.symbol,
                'timestamp': datetime.now().isoformat(),
                'data_period': self.period,
                'total_bars': len(self.data),
                'signals_detected': len(self.signals)
            },
            'current_price': {
                'close': safe_float(current['Close']),
                'open': safe_float(current['Open']),
                'high': safe_float(current['High']),
                'low': safe_float(current['Low']),
                'volume': int(current['Volume']) if not pd.isna(current['Volume']) else 0,
                'change_pct': safe_float(current.get('Price_Change', 0))
            },
            'indicators': {
                'RSI': safe_float(current.get('RSI', None)),
                'MACD': safe_float(current.get('MACD', None)),
                'MACD_Signal': safe_float(current.get('MACD_Signal', None)),
                'ADX': safe_float(current.get('ADX', None)),
                'Stochastic_K': safe_float(current.get('Stoch_K', None)),
                'ATR': safe_float(current.get('ATR', None)),
                'BB_Position': safe_float(current.get('BB_20_Position', None)),
                'Volume_Ratio': safe_float(current['Volume'] / current.get('Volume_MA_20', current['Volume']))
            },
            'moving_averages': {
                'SMA_10': safe_float(current.get('SMA_10', None)),
                'SMA_20': safe_float(current.get('SMA_20', None)),
                'SMA_50': safe_float(current.get('SMA_50', None)),
                'SMA_200': safe_float(current.get('SMA_200', None)),
                'EMA_20': safe_float(current.get('EMA_20', None))
            },
            'signals': self.signals,
            'signal_summary': {
                'total': len(self.signals),
                'bullish': sum(1 for s in self.signals if 'BULLISH' in s['strength']),
                'bearish': sum(1 for s in self.signals if 'BEARISH' in s['strength']),
                'neutral': sum(1 for s in self.signals if 'NEUTRAL' in s['strength']),
                'by_category': {}
            }
        }
        
        # Count by category
        for signal in self.signals:
            cat = signal['category']
            data['signal_summary']['by_category'][cat] = data['signal_summary']['by_category'].get(cat, 0) + 1
        
        # Write JSON with custom encoder
        filename = self.output_dir / f"{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, cls=SafeJSONEncoder)
        
        print(f"✅ JSON saved: {filename}")
        return filename
    
    def export_markdown(self):
        """Export signals to formatted Markdown report"""
        print("📝 Exporting to Markdown...")
        
        current = self.current
        
        def safe_val(val, decimals=2):
            """Safe value formatting"""
            try:
                if pd.isna(val) or np.isinf(val):
                    return "N/A"
                return f"{float(val):.{decimals}f}"
            except:
                return "N/A"
        
        # Build markdown content
        md = f"""# Technical Analysis Report: {self.symbol}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Period:** {self.period}  
**Total Signals:** {len(self.signals)}

---

## 📊 Current Price Data

| Metric | Value |
|--------|-------|
| **Close** | ${safe_val(current['Close'])} |
| **Open** | ${safe_val(current['Open'])} |
| **High** | ${safe_val(current['High'])} |
| **Low** | ${safe_val(current['Low'])} |
| **Volume** | {int(current['Volume']):,} |
| **Change %** | {safe_val(current.get('Price_Change', 0))}% |

---

## 📈 Technical Indicators

| Indicator | Value |
|-----------|-------|
| **RSI(14)** | {safe_val(current.get('RSI', None))} |
| **MACD** | {safe_val(current.get('MACD', None), 4)} |
| **ADX** | {safe_val(current.get('ADX', None))} |
| **Stochastic %K** | {safe_val(current.get('Stoch_K', None))} |
| **ATR** | {safe_val(current.get('ATR', None))} |

### Moving Averages

| Period | SMA | EMA |
|--------|-----|-----|
| **10** | {safe_val(current.get('SMA_10', None))} | {safe_val(current.get('EMA_10', None))} |
| **20** | {safe_val(current.get('SMA_20', None))} | {safe_val(current.get('EMA_20', None))} |
| **50** | {safe_val(current.get('SMA_50', None))} | {safe_val(current.get('EMA_50', None))} |
| **200** | {safe_val(current.get('SMA_200', None))} | {safe_val(current.get('EMA_200', None))} |

---

## 🎯 Detected Signals ({len(self.signals)} Total)

### Signal Summary

"""
        
        # Signal summary by strength
        bullish = sum(1 for s in self.signals if 'BULLISH' in s['strength'])
        bearish = sum(1 for s in self.signals if 'BEARISH' in s['strength'])
        
        md += f"""
| Strength | Count |
|----------|-------|
| 🟢 **Bullish** | {bullish} |
| 🔴 **Bearish** | {bearish} |
| ⚪ **Neutral** | {len(self.signals) - bullish - bearish} |

### Signals by Category

"""
        
        # Count by category
        by_category = {}
        for signal in self.signals:
            cat = signal['category']
            by_category[cat] = by_category.get(cat, 0) + 1
        
        for cat, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
            md += f"- **{cat}**: {count} signals\n"
        
        md += "\n---\n\n### All Signals\n\n"
        
        # Group signals by category
        signals_by_cat = {}
        for signal in self.signals:
            cat = signal['category']
            if cat not in signals_by_cat:
                signals_by_cat[cat] = []
            signals_by_cat[cat].append(signal)
        
        # Write each category
        for cat in sorted(signals_by_cat.keys()):
            md += f"\n#### {cat.replace('_', ' ').title()}\n\n"
            
            for signal in signals_by_cat[cat]:
                strength_emoji = "🟢" if "BULLISH" in signal['strength'] else "🔴" if "BEARISH" in signal['strength'] else "⚪"
                md += f"**{strength_emoji} {signal['signal']}**\n"
                md += f"- {signal['description']}\n"
                md += f"- Strength: {signal['strength']}\n"
                md += f"- Value: {safe_val(signal['value'])}\n\n"
        
        md += "\n---\n\n"
        md += f"*Report generated by YFinance Signal Detector*\n"
        
        # Write markdown file
        filename = self.output_dir / f"{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(filename, 'w') as f:
            f.write(md)
        
        print(f"✅ Markdown saved: {filename}")
        return filename
    
    def run_complete_analysis(self):
        """Run complete pipeline: fetch → calculate → detect → export"""
        print(f"\n{'='*60}")
        print(f"📊 SIGNAL ANALYSIS: {self.symbol}")
        print(f"{'='*60}\n")
        
        # Execute pipeline
        self.fetch_data()
        self.calculate_indicators()
        self.detect_signals()
        
        # Export results
        json_file = self.export_json()
        md_file = self.export_markdown()
        
        print(f"\n{'='*60}")
        print("✅ ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"JSON: {json_file}")
        print(f"MD:   {md_file}")
        print(f"{'='*60}\n")
        
        return {
            'json_file': json_file,
            'md_file': md_file,
            'signals': self.signals
        }


# ============ MAIN EXECUTION ============

if __name__ == "__main__":
    """
    Example usage
    """
    
    # Single stock analysis
    print("\n=== SINGLE STOCK ANALYSIS ===\n")
    detector = SignalDetectorExporter(symbol="AAPL", period="1y")
    results = detector.run_complete_analysis()
    
    print(f"\nDetected {len(results['signals'])} signals")
    print(f"Files saved to: {detector.output_dir}")
    
    
    # Multiple stocks (batch processing)
    print("\n\n=== BATCH ANALYSIS ===\n")
    symbols = ['MSFT', 'GOOGL', 'TSLA']
    
    for symbol in symbols:
        try:
            detector = SignalDetectorExporter(symbol=symbol, period="6mo")
            detector.run_complete_analysis()
        except Exception as e:
            print(f"❌ Error with {symbol}: {str(e)}")
    
    print("\n✅ All analyses complete!")


"""
USAGE EXAMPLES:
===============

1. Basic usage:
   
   from signal_exporter import SignalDetectorExporter
   
   detector = SignalDetectorExporter("AAPL")
   results = detector.run_complete_analysis()

2. Custom output directory:
   
   detector = SignalDetectorExporter("AAPL", output_dir="my_reports")
   results = detector.run_complete_analysis()

3. Different time periods:
   
   detector = SignalDetectorExporter("AAPL", period="3mo")  # 3 months
   detector = SignalDetectorExporter("AAPL", period="2y")   # 2 years
   detector = SignalDetectorExporter("AAPL", period="1d")   # 1 day

4. Access signals programmatically:
   
   detector = SignalDetectorExporter("AAPL")
   results = detector.run_complete_analysis()
   
   for signal in results['signals']:
       print(f"{signal['signal']}: {signal['strength']}")

5. Batch processing with error handling:
   
   symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
   all_results = {}
   
   for symbol in symbols:
       try:
           detector = SignalDetectorExporter(symbol)
           all_results[symbol] = detector.run_complete_analysis()
       except Exception as e:
           print(f"Error with {symbol}: {e}")

OUTPUT FILES:
=============
- JSON: Complete machine-readable data
  - All signals with full details
  - Current price data
  - All indicator values
  - Signal summaries
  
- Markdown: Human-readable report
  - Formatted tables
  - Organized by category
  - Visual indicators (emoji)
  - Easy to read

SAFETY FEATURES:
================
✅ Handles NaN values (returns None or 0)
✅ Handles Inf/-Inf values (returns None)
✅ Safe float conversion for all numeric data
✅ Custom JSON encoder for pandas/numpy types
✅ Validates all data before serialization
✅ No data can break JSON structure
✅ Error handling for edge cases
"""