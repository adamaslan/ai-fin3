# 📊 Easy Technical Signal Scanner - Documentation

## 📋 Overview

The **Easy Technical Signal Scanner** is a user-friendly wrapper around a comprehensive technical analysis system that scans for 150+ technical signals with Fibonacci levels and optional AI-powered ranking.

## 🚀 Quick Start

### Option 1: Interactive Mode (Easiest)
```python
from easy_scanner import scan_interactive

analyzer = scan_interactive()
```

### Option 2: Direct Mode (Quickest)
```python
from easy_scanner import scan
import os

analyzer = scan('AAPL', '1y', os.getenv('GEMINI_API_KEY'))
```

### Option 3: Advanced Mode (Most Control)
```python
from easy_scanner import EasySignalScanner
import os

scanner = EasySignalScanner(
    symbol='TSLA', 
    period='6mo', 
    gemini_api_key=os.getenv('GEMINI_API_KEY')
)
analyzer = scanner.run_analysis()
```

## ⏰ Supported Time Periods

- `1d` - 1 day
- `5d` - 5 days
- `1mo` - 1 month
- `3mo` - 3 months
- `6mo` - 6 months
- `1y` - 1 year (default)
- `2y` - 2 years
- `5y` - 5 years
- `max` - Maximum available

## 🔍 Signal Categories

The scanner detects signals across 10+ technical analysis categories:

### 1. **Moving Averages (MA)**
- **Golden Cross** (50D MA > 200D MA)
- **Death Cross** (50D MA < 200D MA)
- **Price Crosses** above/below key MAs
- **MA Convergence/Divergence** patterns

### 2. **Fibonacci Levels**
- **Support/Resistance** at key Fibonacci levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
- **Golden Ratio** (61.8%) breakouts/breakdowns
- **Fibonacci Extensions** for price targets

### 3. **Oscillators & Momentum**
- **RSI Divergence** (bullish/bearish)
- **RSI Overbought/Oversold** (above 70/below 30)
- **MACD Crossovers**
- **MACD Divergence**
- **Stochastic** signals
- **Williams %R** extremes

### 4. **Volume Indicators**
- **Volume Surge** (abnormal volume)
- **On-Balance Volume (OBV)** trends
- **Volume-Price Trends**
- **Accumulation/Distribution**

### 5. **Volatility Indicators**
- **Bollinger Band** squeezes/expansions
- **Bollinger Band** breakouts
- **ATR** (Average True Range) expansion
- **Volatility Breakouts**

### 6. **Trend Indicators**
- **ADX** (Average Directional Index) strength
- **Ichimoku Cloud** signals
- **Parabolic SAR** reversals
- **Trend Line** breaks

### 7. **Support & Resistance**
- **Horizontal S/R** breaks
- **Pivot Point** levels
- **Round Number** psychological levels

### 8. **Pattern Recognition**
- **Double Top/Bottom**
- **Head & Shoulders**
- **Triangle** formations
- **Flag/Pennant** patterns

### 9. **Candlestick Patterns**
- **Doji** (indecision)
- **Hammer/Hanging Man**
- **Engulfing Patterns**
- **Morning/Evening Star**

### 10. **Market Structure**
- **Higher Highs/Lower Lows**
- **Breakout/Breakdown** from consolidation
- **Gap** fills/continuations

## 📊 Signal Strength Classification

Each signal is classified by strength:

### **🔴 CRITICAL** (Highest Priority)
- Major trend reversals
- High-volume breakouts
- Multiple indicator confluence

### **🟠 STRONG** (High Confidence)
- Clear pattern completions
- Strong momentum shifts
- Key level breaks

### **🟡 MODERATE** (Medium Confidence)
- Single indicator signals
- Early pattern formations
- Minor level tests

### **🟢 WEAK** (Low Priority)
- Minor divergences
- Oversold/overbought rebounds
- Noise signals

## 🤖 AI Signal Ranking (Optional)

When Gemini API key is provided, signals are ranked by:

### **AI Score (0-100)**
- `90-100`: Very High Confidence
- `75-89`: High Confidence
- `60-74`: Medium Confidence
- `40-59`: Low Confidence
- `0-39`: Very Low Confidence

### **AI Analysis Factors**
1. **Signal Confluence** - Multiple indicators agreeing
2. **Volume Confirmation** - Volume supporting price action
3. **Trend Alignment** - Signals in direction of primary trend
4. **Historical Reliability** - Pattern success rate
5. **Market Context** - Overall market conditions

## 📈 Output Data Structure

### **Signal Object Example**
```python
{
    "signal": "GOLDEN_CROSS_50D_200D",
    "category": "MA_CROSSOVER",
    "strength": "BULLISH_STRONG",
    "ai_score": 85.5,  # If AI enabled
    "description": "50-day MA crossed above 200-day MA",
    "impact": "LONG_TERM_TREND_REVERSAL",
    "timeframe": "DAILY",
    "value": "50D MA: 150.25, 200D MA: 148.75",
    "last_occurrence": "2024-01-15",
    "occurrence_count": 3
}
```

### **Technical Indicators Calculated**
```python
{
    "RSI": 65.2,                    # Relative Strength Index
    "MACD": 1.25,                   # MACD line
    "MACD_Signal": 0.98,            # Signal line
    "MACD_Histogram": 0.27,         # Histogram
    "ADX": 35.4,                    # Average Directional Index
    "ATR": 2.15,                    # Average True Range
    "Volatility": 18.3,             # Volatility percentage
    "BB_Upper": 155.25,             # Bollinger Band upper
    "BB_Lower": 145.75,             # Bollinger Band lower
    "SMA_20": 150.50,               # 20-day Simple MA
    "SMA_50": 148.75,               # 50-day Simple MA
    "SMA_200": 146.25,              # 200-day Simple MA
    "Volume_Avg": 4500000,          # Average volume
    "Volume_Current": 6200000,      # Current volume
    "Price_Change": 2.5,            # Daily price change %
    "Fibonacci_23_6": 142.30,       # Fibonacci 23.6% level
    "Fibonacci_38_2": 140.15,       # Fibonacci 38.2% level
    "Fibonacci_50_0": 138.45,       # Fibonacci 50% level
    "Fibonacci_61_8": 136.75,       # Fibonacci 61.8% level
    "Fibonacci_78_6": 134.90        # Fibonacci 78.6% level
}
```

## 💾 Export Options

### **1. CSV Export**
```python
scanner.export_csv()  # Default filename
# or
scanner.export_csv("custom_filename.csv")
```

### **2. JSON Export**
```python
scanner.export_json()  # Default filename
# or
scanner.export_json("custom_filename.json")
```

### **3. DataFrame Access**
```python
signals_df = scanner.get_signals_dataframe()
print(signals_df.head(10))
```

## 🔧 Filtering Methods

### **By Category**
```python
fib_signals = scanner.get_fibonacci_signals()
ma_signals = scanner.get_ma_signals()
rsi_signals = scanner.filter_by_category("RSI")
```

### **By AI Score**
```python
high_conf_signals = scanner.filter_by_score(75)  # ≥75 AI score
very_high_conf = scanner.filter_by_score(90)     # ≥90 AI score
```

### **By Strength**
```python
critical_signals = [s for s in analyzer.signals 
                    if "CRITICAL" in s['strength']]
bullish_signals = [s for s in analyzer.signals 
                   if "BULLISH" in s['strength']]
```

## 📊 Sample Output Summary

```
📊 ANALYSIS SUMMARY
================================================================================
Symbol: AAPL | Period: 1y
Price: $185.64 | Change: +1.25%
Date: 2024-01-15 16:00:00

📈 Technical Indicators:
   RSI: 62.1 | MACD: 1.245 | ADX: 28.5
   Volatility: 18.3% | ATR: 2.15

🎯 Signal Overview:
   Total Signals: 24
   Bullish: 14 | Bearish: 8 | Neutral: 2
   AI Ranked: ✅ (Average Score: 72.4)
   Bias: 🟢 BULLISH

🔝 Top 5 Signals:
1. GOLDEN_CROSS_50D_200D (AI: 87) - 🟢 BULLISH_STRONG
2. RSI_BULLISH_DIVERGENCE (AI: 82) - 🟢 BULLISH_MODERATE
3. FIBONACCI_61_8_SUPPORT (AI: 79) - 🟢 BULLISH_WEAK
4. VOLUME_SURGE_UP (AI: 76) - 🟢 BULLISH_MODERATE
5. BOLLINGER_BAND_SQUEEZE (AI: 74) - 🟡 NEUTRAL_MODERATE

📂 Results saved to: /data/analysis/20240115_160000_AAPL/
================================================================================
```

## 📁 File Structure

```
data/
└── analysis/
    └── YYYYMMDD_HHMMSS_SYMBOL/
        ├── signals.csv              # All signals with details
        ├── technical_data.csv       # Full indicator data
        ├── summary.txt              # Text summary
        ├── signals.json             # JSON export
        └── charts/                  # Generated charts (if enabled)
```

## ⚡ Performance Notes

- **Data Source**: Yahoo Finance (free, real-time)
- **Processing Time**: ~2-5 seconds per symbol
- **AI Ranking**: Adds ~3-5 seconds with Gemini API
- **Memory Usage**: Minimal (<100MB per analysis)
- **Rate Limits**: None for basic analysis

## 🔐 API Key Setup (Optional)

```bash
# Set Gemini API key for AI ranking
export GEMINI_API_KEY="your-api-key-here"
```

## 🚨 Common Signals to Watch

### **Bullish Signals**
1. **Golden Cross** - Long-term bullish trend
2. **RSI Bullish Divergence** - Momentum reversal
3. **Volume Surge on Up Days** - Institutional buying
4. **Break Above Fibonacci 61.8%** - Strong momentum
5. **Bollinger Band Squeeze Breakout** - Volatility expansion

### **Bearish Signals**
1. **Death Cross** - Long-term bearish trend
2. **RSI Bearish Divergence** - Momentum loss
3. **Volume Surge on Down Days** - Institutional selling
4. **Break Below Fibonacci 38.2%** - Weakness
5. **Head & Shoulders Pattern** - Trend reversal

### **Neutral/Consolidation Signals**
1. **Doji Candles** - Indecision
2. **Low ADX (<25)** - Weak/no trend
3. **Inside Bars** - Consolidation
4. **Triangle Patterns** - Continuation preparation

## 📚 Best Practices

1. **Timeframe Selection**: Use longer periods (1y+) for trend analysis
2. **Signal Confirmation**: Wait for multiple signals in agreement
3. **Volume Check**: Always verify with volume confirmation
4. **AI Ranking**: Use for signal prioritization, not absolute decisions
5. **Backtesting**: Test signals against historical performance

## 🔄 Update Frequency

- **Intraday**: Re-run every 4 hours for active trading
- **Daily**: End-of-day analysis for position planning
- **Weekly**: Sunday night for weekly outlook

This scanner provides comprehensive technical analysis with an emphasis on usability and actionable signals, making sophisticated analysis accessible to all levels of traders.