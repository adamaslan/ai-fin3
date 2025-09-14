# organized just rule based minimum
# Cell 1: Imports and Setup
import requests
import json
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import date, datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load environment variables and setup device
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# part 2 
# 7 Current Technical Indicators Fetcher
import requests
import pandas as pd
import json
from datetime import datetime
import time

# Your Alpha Vantage API Key
# ALPHA_VANTAGE_API_KEY = "ALPHA_VANTAGE_API_KEY"  # Replace with your actual API key

def fetch_indicator_data(params, indicator_name):
    """Fetch technical indicator data from Alpha Vantage API"""
    try:
        url = "https://www.alphavantage.co/query"
        response = requests.get(url, params=params)
        data = response.json()
        
        # Check for API errors
        if "Error Message" in data:
            print(f"❌ Error fetching {indicator_name}: {data['Error Message']}")
            return None
        elif "Note" in data:
            print(f"⚠️  API limit reached for {indicator_name}: {data['Note']}")
            return None
        elif "Information" in data:
            print(f"ℹ️  {indicator_name}: {data['Information']}")
            return None
            
        # Get the technical analysis data
        tech_key = None
        for key in data.keys():
            if "Technical Analysis" in key:
                tech_key = key
                break
                
        if not tech_key:
            print(f"❌ No technical analysis data found for {indicator_name}")
            return None
            
        tech_data = data[tech_key]
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(tech_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index(ascending=False)  # Most recent first
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
        
    except Exception as e:
        print(f"❌ Error fetching {indicator_name}: {str(e)}")
        return None

def fetch_current_price(symbol):
    """Fetch current price using Alpha Vantage"""
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        url = "https://www.alphavantage.co/query"
        response = requests.get(url, params=params)
        data = response.json()
        
        if "Global Quote" in data:
            quote = data["Global Quote"]
            current_price = float(quote["05. price"])
            change = float(quote["09. change"])
            change_percent = quote["10. change percent"].replace("%", "")
            
            return {
                "price": current_price,
                "change": change,
                "change_percent": float(change_percent)
            }
    except Exception as e:
        print(f"❌ Error fetching current price: {str(e)}")
        return None

def get_current_technical_indicators(symbol):
    """Fetch current levels of all technical indicators"""
    
    print(f"🔍 Fetching Current Technical Indicators for {symbol}")
    print("=" * 60)
    
    # Store all current values
    current_indicators = {
        "symbol": symbol,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "indicators": {}
    }
    
    # 1. Current Price
    print("📊 Fetching Current Price...")
    price_data = fetch_current_price(symbol)
    if price_data:
        current_indicators["current_price"] = price_data
        print(f"✅ Current Price: ${price_data['price']:.2f} ({price_data['change']:+.2f}, {price_data['change_percent']:+.2f}%)")
    
    # 2. EMA 10
    print("\n📈 Fetching 10-period EMA...")
    ema10_params = {
        "function": "EMA",
        "symbol": symbol,
        "interval": "daily",
        "time_period": "10",
        "series_type": "close",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    ema10_data = fetch_indicator_data(ema10_params, "EMA 10")
    if ema10_data is not None and not ema10_data.empty:
        current_ema10 = ema10_data.iloc[0]["EMA"]
        current_indicators["indicators"]["EMA_10"] = current_ema10
        print(f"✅ EMA 10: ${current_ema10:.2f}")
    
    time.sleep(12)  # API rate limit protection
    
    # 3. EMA 20
    print("\n📈 Fetching 20-period EMA...")
    ema20_params = {
        "function": "EMA",
        "symbol": symbol,
        "interval": "daily",
        "time_period": "20",
        "series_type": "close",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    ema20_data = fetch_indicator_data(ema20_params, "EMA 20")
    if ema20_data is not None and not ema20_data.empty:
        current_ema20 = ema20_data.iloc[0]["EMA"]
        current_indicators["indicators"]["EMA_20"] = current_ema20
        print(f"✅ EMA 20: ${current_ema20:.2f}")
    
    time.sleep(12)
    
    # 4. SMA 50
    print("\n📏 Fetching 50-period SMA...")
    sma50_params = {
        "function": "SMA",
        "symbol": symbol,
        "interval": "daily",
        "time_period": "50",
        "series_type": "close",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    sma50_data = fetch_indicator_data(sma50_params, "SMA 50")
    if sma50_data is not None and not sma50_data.empty:
        current_sma50 = sma50_data.iloc[0]["SMA"]
        current_indicators["indicators"]["SMA_50"] = current_sma50
        print(f"✅ SMA 50: ${current_sma50:.2f}")
    
    time.sleep(12)
    
    # 5. RSI
    print("\n⚡ Fetching RSI...")
    rsi_params = {
        "function": "RSI",
        "symbol": symbol,
        "interval": "daily",
        "time_period": "14",
        "series_type": "close",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    rsi_data = fetch_indicator_data(rsi_params, "RSI")
    if rsi_data is not None and not rsi_data.empty:
        current_rsi = rsi_data.iloc[0]["RSI"]
        current_indicators["indicators"]["RSI"] = current_rsi
        
        # RSI interpretation
        if current_rsi > 70:
            rsi_signal = "OVERBOUGHT 🔴"
        elif current_rsi < 30:
            rsi_signal = "OVERSOLD 🟢"
        else:
            rsi_signal = "NEUTRAL 🟡"
            
        print(f"✅ RSI: {current_rsi:.2f} ({rsi_signal})")
    
    time.sleep(12)
    
    # 6. MACD
    print("\n📊 Fetching MACD...")
    macd_params = {
        "function": "MACD",
        "symbol": symbol,
        "interval": "daily",
        "series_type": "close",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    macd_data = fetch_indicator_data(macd_params, "MACD")
    if macd_data is not None and not macd_data.empty:
        current_macd = macd_data.iloc[0]["MACD"]
        current_signal = macd_data.iloc[0]["MACD_Signal"]
        current_hist = macd_data.iloc[0]["MACD_Hist"]
        
        current_indicators["indicators"]["MACD"] = current_macd
        current_indicators["indicators"]["MACD_Signal"] = current_signal
        current_indicators["indicators"]["MACD_Hist"] = current_hist
        
        # MACD interpretation
        if current_macd > current_signal:
            macd_signal = "BULLISH 🟢"
        else:
            macd_signal = "BEARISH 🔴"
            
        print(f"✅ MACD: {current_macd:.4f}")
        print(f"   Signal: {current_signal:.4f}")
        print(f"   Histogram: {current_hist:.4f} ({macd_signal})")
    
    time.sleep(12)
    
    # 7. Bollinger Bands
    print("\n📊 Fetching Bollinger Bands...")
    bb_params = {
        "function": "BBANDS",
        "symbol": symbol,
        "interval": "daily",
        "time_period": "20",
        "series_type": "close",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    bb_data = fetch_indicator_data(bb_params, "Bollinger Bands")
    if bb_data is not None and not bb_data.empty:
        current_upper = bb_data.iloc[0]["Real Upper Band"]
        current_middle = bb_data.iloc[0]["Real Middle Band"]
        current_lower = bb_data.iloc[0]["Real Lower Band"]
        
        current_indicators["indicators"]["BB_Upper"] = current_upper
        current_indicators["indicators"]["BB_Middle"] = current_middle
        current_indicators["indicators"]["BB_Lower"] = current_lower
        
        print(f"✅ Bollinger Bands:")
        print(f"   Upper: ${current_upper:.2f}")
        print(f"   Middle: ${current_middle:.2f}")
        print(f"   Lower: ${current_lower:.2f}")
        
        # BB position analysis
        if price_data:
            current_price = price_data["price"]
            if current_price > current_upper:
                bb_position = "ABOVE UPPER (Overbought) 🔴"
            elif current_price < current_lower:
                bb_position = "BELOW LOWER (Oversold) 🟢"
            else:
                bb_position = "WITHIN BANDS (Normal) 🟡"
            print(f"   Position: {bb_position}")
    
    # Summary Analysis
    print(f"\n\n🎯 CURRENT TECHNICAL ANALYSIS SUMMARY FOR {symbol}")
    print("=" * 60)
    
    if price_data and "EMA_10" in current_indicators["indicators"] and "EMA_20" in current_indicators["indicators"]:
        current_price = price_data["price"]
        ema10 = current_indicators["indicators"]["EMA_10"]
        ema20 = current_indicators["indicators"]["EMA_20"]
        
        print(f"💰 Current Price: ${current_price:.2f}")
        print(f"📈 EMA 10: ${ema10:.2f}")
        print(f"📈 EMA 20: ${ema20:.2f}")
        
        # EMA Crossover Analysis
        if ema10 > ema20:
            ema_signal = "BULLISH (10 EMA > 20 EMA) 🟢"
        else:
            ema_signal = "BEARISH (10 EMA < 20 EMA) 🔴"
        print(f"🔄 EMA Signal: {ema_signal}")
        
        # Price vs EMAs
        if current_price > ema10 > ema20:
            trend_signal = "STRONG UPTREND 🚀"
        elif current_price < ema10 < ema20:
            trend_signal = "STRONG DOWNTREND 📉"
        else:
            trend_signal = "MIXED/CONSOLIDATION ⚖️"
        print(f"📊 Trend Signal: {trend_signal}")
    
    if "SMA_50" in current_indicators["indicators"]:
        sma50 = current_indicators["indicators"]["SMA_50"]
        print(f"📏 SMA 50: ${sma50:.2f}")
        
        if price_data:
            if current_price > sma50:
                ma50_signal = "ABOVE 50 MA (Bullish) 🟢"
            else:
                ma50_signal = "BELOW 50 MA (Bearish) 🔴"
            print(f"🎯 50 MA Signal: {ma50_signal}")
    
    print(f"\n📅 Analysis Time: {current_indicators['timestamp']}")
    
    return current_indicators

# Main execution
if __name__ == "__main__":
    # Allow symbol to be configurable instead of hardcoded
    SYMBOL = 'NVDA'  # Default symbol
    
    # You can change this to accept command line arguments if needed
    # import sys
    # SYMBOL = sys.argv[1] if len(sys.argv) > 1 else 'NVDA'
    
    # Fetch all current technical indicators
    current_data = get_current_technical_indicators(SYMBOL)
    
    print(f"\n💾 Data saved for {SYMBOL}")
    print("Raw data structure:")
    print(json.dumps(current_data, indent=2, default=str))

    # Execute the spread analysis
    try:
        # Try to use the variable from your main execution
        spread_analysis = suggest_vertical_spreads(current_data)
    except NameError:
        print(f"🚀 Please run the technical indicators fetcher for {SYMBOL} first!")
        print("Then this cell will automatically use the 'current_data' variable.")
        print("")
        print("Or manually run:")
        print("spread_analysis = suggest_vertical_spreads(current_data)")

# Update the server section to use the dynamic SYMBOL
print(f"🔥 Starting {SYMBOL} Trading API Server")
print("📡 API will be available at: http://localhost:8000")
print("📖 API docs available at: http://localhost:8000/docs")
print(f"🔄 WebSocket endpoint: ws://localhost:8000/ws/{SYMBOL}")
# print(f"✅ Updated trading data for {SYMBOL}")
# print(f"🎯 VERTICAL SPREAD SUGGESTIONS FOR {current_data['symbol']}")
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def suggest_vertical_spreads(current_data):
    """
    Suggest vertical spreads based on technical indicator analysis
    """
    print(f"🎯 VERTICAL SPREAD SUGGESTIONS FOR {current_data['symbol']}")
    print("=" * 70)
    
    # Extract key data points
    current_price = current_data.get("current_price", {}).get("price", 0)
    indicators = current_data.get("indicators", {})
    
    # Key technical levels
    ema10 = indicators.get("EMA_10", 0)
    ema20 = indicators.get("EMA_20", 0)
    sma50 = indicators.get("SMA_50", 0)
    rsi = indicators.get("RSI", 50)
    bb_upper = indicators.get("BB_Upper", 0)
    bb_lower = indicators.get("BB_Lower", 0)
    bb_middle = indicators.get("BB_Middle", 0)
    macd = indicators.get("MACD", 0)
    macd_signal = indicators.get("MACD_Signal", 0)
    
    print(f"📊 Current Analysis:")
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   EMA 10: ${ema10:.2f}")
    print(f"   EMA 20: ${ema20:.2f}")
    print(f"   SMA 50: ${sma50:.2f}")
    print(f"   RSI: {rsi:.2f}")
    print(f"   BB Upper: ${bb_upper:.2f}")
    print(f"   BB Lower: ${bb_lower:.2f}")
    print(f"   MACD: {macd:.4f}")
    
    # Determine market bias
    bullish_signals = 0
    bearish_signals = 0
    
    # EMA Analysis
    if ema10 > ema20:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    # Price vs EMAs
    if current_price > ema10:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    # RSI Analysis
    if rsi < 30:
        bullish_signals += 1  # Oversold = potential bounce
    elif rsi > 70:
        bearish_signals += 1  # Overbought = potential pullback
    
    # MACD Analysis
    if macd > macd_signal:
        bullish_signals += 1
    else:
        bearish_signals += 1
    
    # Bollinger Band Analysis
    if current_price < bb_lower:
        bullish_signals += 1  # Oversold
    elif current_price > bb_upper:
        bearish_signals += 1  # Overbought
    
    # Determine overall bias
    if bullish_signals > bearish_signals:
        market_bias = "BULLISH"
        bias_strength = bullish_signals / (bullish_signals + bearish_signals)
    else:
        market_bias = "BEARISH"
        bias_strength = bearish_signals / (bullish_signals + bearish_signals)
    
    print(f"\n🎯 Market Bias: {market_bias} (Strength: {bias_strength:.1%})")
    print(f"   Bullish Signals: {bullish_signals}")
    print(f"   Bearish Signals: {bearish_signals}")
    
    # Calculate support and resistance levels
    support_levels = []
    resistance_levels = []
    
    # Add EMA levels
    support_levels.extend([ema10, ema20, sma50])
    resistance_levels.extend([ema10, ema20, sma50])
    
    # Add Bollinger Band levels
    support_levels.append(bb_lower)
    resistance_levels.append(bb_upper)
    
    # Filter and sort levels
    support_levels = sorted([level for level in support_levels if level > 0 and level < current_price])
    resistance_levels = sorted([level for level in resistance_levels if level > current_price])
    
    # Calculate expiration dates
    today = datetime.now()
    expirations = {
        "1 Week": today + timedelta(days=7),
        "2 Weeks": today + timedelta(days=14),
        "4 Weeks": today + timedelta(days=28),
        "6 Weeks": today + timedelta(days=42)
    }
    
    print(f"\n📅 Expiration Dates:")
    for period, date in expirations.items():
        # Find next Friday
        days_until_friday = (4 - date.weekday()) % 7
        if days_until_friday == 0 and date.hour >= 16:  # If it's Friday after market close
            days_until_friday = 7
        friday_date = date + timedelta(days=days_until_friday)
        print(f"   {period}: {friday_date.strftime('%Y-%m-%d (%A)')}")
    
    print(f"\n" + "=" * 70)
    print(f"📊 VERTICAL SPREAD RECOMMENDATIONS")
    print(f"=" * 70)
    
    # Generate spread suggestions for each timeframe
    for period, exp_date in expirations.items():
        print(f"\n⏰ {period.upper()} EXPIRATION ({exp_date.strftime('%m/%d/%Y')})")
        print("-" * 50)
        
        # Calculate expected price range based on timeframe
        days_to_exp = (exp_date - today).days
        
        # Volatility estimate (simplified)
        if bb_upper > 0 and bb_lower > 0:
            implied_volatility = (bb_upper - bb_lower) / bb_middle
        else:
            implied_volatility = 0.20  # Default 20%
        
        # Expected move calculation
        expected_move = current_price * implied_volatility * np.sqrt(days_to_exp / 365)
        
        print(f"Expected Move: ±${expected_move:.2f}")
        print(f"Price Range: ${current_price - expected_move:.2f} - ${current_price + expected_move:.2f}")
        
        # CALL CREDIT SPREADS (Bearish/Neutral)
        print(f"\n📈 CALL CREDIT SPREADS (Short Call Spread)")
        
        # Strike selection based on technical levels
        if resistance_levels:
            short_call_strike = min(resistance_levels)
        else:
            short_call_strike = current_price + expected_move * 0.5
        
        long_call_strike = short_call_strike + (current_price * 0.05)  # 5% wide
        
        # Adjust strikes to reasonable increments
        short_call_strike = round(short_call_strike * 2) / 2  # Round to nearest $0.50
        long_call_strike = round(long_call_strike * 2) / 2
        
        print(f"   Suggested Strike: Sell ${short_call_strike:.2f} Call / Buy ${long_call_strike:.2f} Call")
        print(f"   Max Profit: Premium Collected")
        print(f"   Max Loss: ${long_call_strike - short_call_strike:.2f} - Premium")
        print(f"   Breakeven: ${short_call_strike:.2f} + Premium")
        
        # Technical justification
        if short_call_strike > bb_upper:
            print(f"   📊 Technical: Above Bollinger Upper Band (${bb_upper:.2f})")
        if short_call_strike > ema10:
            print(f"   📊 Technical: Above EMA 10 (${ema10:.2f})")
        if rsi > 60:
            print(f"   📊 Technical: RSI suggests potential resistance at {rsi:.1f}")
        
        # PUT CREDIT SPREADS (Bullish/Neutral)
        print(f"\n📉 PUT CREDIT SPREADS (Short Put Spread)")
        
        # Strike selection based on technical levels
        if support_levels:
            short_put_strike = max(support_levels)
        else:
            short_put_strike = current_price - expected_move * 0.5
        
        long_put_strike = short_put_strike - (current_price * 0.05)  # 5% wide
        
        # Adjust strikes to reasonable increments
        short_put_strike = round(short_put_strike * 2) / 2  # Round to nearest $0.50
        long_put_strike = round(long_put_strike * 2) / 2
        
        print(f"   Suggested Strike: Sell ${short_put_strike:.2f} Put / Buy ${long_put_strike:.2f} Put")
        print(f"   Max Profit: Premium Collected")
        print(f"   Max Loss: ${short_put_strike - long_put_strike:.2f} - Premium")
        print(f"   Breakeven: ${short_put_strike:.2f} - Premium")
        
        # Technical justification
        if short_put_strike < bb_lower:
            print(f"   📊 Technical: Below Bollinger Lower Band (${bb_lower:.2f})")
        if short_put_strike > ema20:
            print(f"   📊 Technical: Above EMA 20 support (${ema20:.2f})")
        if rsi < 40:
            print(f"   📊 Technical: RSI suggests potential support at {rsi:.1f}")
        
        # Risk Management
        print(f"\n⚠️  RISK MANAGEMENT:")
        print(f"   • Close at 50% max profit")
        print(f"   • Close at 21 DTE if not profitable")
        print(f"   • Monitor key technical levels")
        print(f"   • Consider early assignment risk")
    
    # Overall Strategy Recommendation
    print(f"\n" + "=" * 70)
    print(f"🎯 OVERALL STRATEGY RECOMMENDATION")
    print(f"=" * 70)
    
    if market_bias == "BULLISH":
        print(f"💡 Primary Focus: PUT CREDIT SPREADS")
        print(f"   Rationale: Bullish bias suggests selling puts below support")
        print(f"   Secondary: Conservative call spreads well above resistance")
    else:
        print(f"💡 Primary Focus: CALL CREDIT SPREADS")
        print(f"   Rationale: Bearish bias suggests selling calls above resistance")
        print(f"   Secondary: Conservative put spreads near strong support")
    
    print(f"\n📊 Key Levels to Watch:")
    if support_levels:
        print(f"   Support: ${max(support_levels):.2f}")
    if resistance_levels:
        print(f"   Resistance: ${min(resistance_levels):.2f}")
    print(f"   Current Price: ${current_price:.2f}")
    
    return {
        "market_bias": market_bias,
        "bias_strength": bias_strength,
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "expected_moves": expected_move
    }

# Execute the spread analysis
if __name__ == "__main__":
    # This assumes you have the current_data from the previous cell
    # You would run this after executing the technical indicators fetcher
    
    # Check if current_data exists from previous cell
    try:
        # Try to use the variable from your main execution
        spread_analysis = suggest_vertical_spreads(current_data)
    except NameError:
        print("🚀 Please run the technical indicators fetcher first!")
        print("Then this cell will automatically use the 'current_data' variable.")
        print("")
        print("Or manually run:")
        print("spread_analysis = suggest_vertical_spreads(current_data)")

        import threading
import nest_asyncio

nest_asyncio.apply()

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
# 10

import uvicorn



print("🔥 Starting QUBT Trading API Server")
print("📡 API will be available at: http://localhost:8000")
print("📖 API docs available at: http://localhost:8000/docs")
print("🔄 WebSocket endpoint: ws://localhost:8000/ws/QUBT")
# print(f"✅ Updated trading data for {QUBT}")
# print(f"🎯 VERTICAL SPREAD SUGGESTIONS FOR {current_data['QUBT']}")
uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


# Start server in background
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

print("🔥 Server started! Check http://localhost:8000")
