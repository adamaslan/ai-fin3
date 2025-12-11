# Breakdown of boll4-nov-g.py

This document breaks down the `boll4-nov-g.py` script into its core components, explaining the purpose of each section.

## 1. Imports and API Key Management

This section covers the initial library imports and the logic for securely managing the Gemini API key. The script first tries to load the key from environment variables (or a `.env` file) and, if not found, interactively prompts the user to enter it.

```python
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

def get_gemini_api_key():
    """
    Retrieves the Gemini API key from environment variables or prompts the user.
    """
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("GEMINI_API_KEY environment variable not found.")
        try:
            import getpass
            api_key = getpass.getpass("Please enter your Gemini API key: ")
        except (ImportError, ModuleNotFoundError):
            # Fallback for environments where getpass is not available
            api_key = input("Please enter your Gemini API key: ")
    return api_key

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
```

## 2. Technical Indicators Calculation

This is the core logic for calculating over 50 technical indicators. The `calculate_indicators` method orchestrates the process, calling various private helper methods (`_calculate_atr`, `_calculate_adx`, etc.) to compute values for RSI, MACD, Bollinger Bands, and many others.

```python
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
        
        # ... (and many more indicator calculations) ...

        self.data = df
        print("✅ All 50+ indicators calculated")
        return df

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
        # ... implementation ...
        return df
    
    # ... other private calculation helpers ...
```

## 3. Signal Detection (200+ Signals)

The `detect_signals` method iterates through the historical data and applies a vast set of rules to identify over 200 distinct bullish, bearish, and neutral trading signals. These signals cover everything from moving average crossovers and RSI divergences to candlestick patterns and volume spikes.

```python
    def detect_signals(self):
        """Detect 200+ comprehensive technical signals"""
        df = self.data.copy()
        current = df.iloc[-1]
        prev = df.iloc[-2]
        # ...
        
        signals = []
        
        print("\n🎯 Scanning for 200+ Technical Alerts...")
        
        # ============ MOVING AVERAGE SIGNALS (40 signals) =============
        
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
        
        # ... (many more signal detection rules) ...

        # ============ RSI SIGNALS (30 signals) ============ 
        # ...

        # ============ MACD SIGNALS (25 signals) ============ 
        # ...

        self.signals = signals
        print(f"✅ Detected {len(signals)} Active Signals")
        return signals
```

## 4. AI Prompt Construction

This section details how the script communicates with the Gemini AI. It constructs detailed prompts that include market data, technical indicators, and detected signals. It asks the AI to perform several tasks: rank the signals by importance, provide a detailed market analysis, and generate specific trade recommendations.

```python
    def analyze_with_ai(self):
        """Multi-stage AI analysis"""
        # ...
        # Stage 1: Rank all signals
        print("  📊 Stage 1: Ranking signals...")
        self._rank_signals_with_ai()
        
        # Stage 2: Deep market analysis
        print("  🔍 Stage 2: Deep market analysis...")
        market_analysis = self._get_market_analysis(current, analysis)
        
        # Stage 3: Options strategy recommendations
        print("  💰 Stage 3: Generating trade recommendations...")
        trade_recommendations = self._get_trade_recommendations(current, analysis)
        # ...

    def _rank_signals_with_ai(self):
        """Rank signals with AI scoring"""
        # ...
        prompt = f"""Score these trading signals for {self.symbol} (Price: ${current['Close']:.2f}).
Score each 1-100 based on: reliability, timing, risk/reward, actionability.
# ...

    def _get_market_analysis(self, current, analysis):
        """Get comprehensive market analysis from AI"""
        # ...
        prompt = f"""Create a detailed, article-style investment analysis for {self.symbol}.
The output must be a single, valid JSON object.
# ...

    def _get_trade_recommendations(self, current, analysis):
        """Get specific trade recommendations"""
        # ...
        prompt = f"""Generate specific trade recommendations for {self.symbol}.
# ...
```

## 5. JSON Object Creation

The script creates several structured JSON objects for analysis and storage. The primary one is generated by `generate_comprehensive_analysis`, which consolidates price data, indicator ratings, signal summaries, and key price levels into a single object. Other methods format data for local saving and for upload to Google Cloud Storage.

```python
    def generate_comprehensive_analysis(self):
        """Generate complete analysis with indicator ratings"""
        current = self.data.iloc[-1]
        
        indicator_ratings = {
            'trend_strength': self._rate_trend_strength(current),
            'momentum': self._rate_momentum(current),
            # ...
        }
        
        analysis = {
            'symbol': self.symbol,
            'timestamp': datetime.now().isoformat(),
            'price_data': {
                'current_price': float(current['Close']),
                # ...
            },
            'indicator_ratings': indicator_ratings,
            'overall_bias': overall_bias,
            'bias_score': bias_score,
            'signal_summary': {
                # ...
            },
            'key_levels': self._identify_key_levels(current),
            'signals': self.signals
        }
        
        return analysis

    def save_locally(self):
        """Save comprehensive data locally"""
        # ...
        analysis = self.generate_comprehensive_analysis()
        # ...
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        # ...
```

## 6. Execution and Configuration

This is the entry point of the script. The `main` function sets the configuration (like the stock symbol), initializes the `AdvancedTechnicalAnalyzer` class, and runs the entire analysis pipeline from data fetching to saving the results.

```python
def main():
    """Main execution function to run the advanced technical analysis."""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # --- Configuration ---
    SYMBOL = 'QQQ'
    PERIOD = '1y'
    UPLOAD_TO_GCP = True
    GEMINI_API_KEY = get_gemini_api_key()
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
        if GEMINI_API_KEY:
            analyzer.ai_analysis = analyzer.analyze_with_ai()
        else:
            print("   ⚠️  No API key - skipping AI analysis")
        
        print("\n💾 Saving results...")
        analyzer.save_locally()
        
        if UPLOAD_TO_GCP:
            try:
                analyzer.save_to_gcp()
            except Exception as e:
                print(f"   ⚠️  GCP upload skipped: {str(e)[:50]}")
        
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()


# Example usage
if __name__ == "__main__":
    main()
```