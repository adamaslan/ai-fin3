"""
AI-Enhanced Options Spread Analysis with GCP Storage
Mistral AI + yfinance + Google Cloud Storage
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import os
import time
import json
from dotenv import load_dotenv
from mistralai import Mistral
from google.cloud import storage

# --- CONFIGURATION ---
TICKER = "MP" 
DAYS_OF_HISTORY = 90
GCP_BUCKET = 'ttb-bucket1'

# 1. Load environment variables from a .env file if one exists
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Look for .env in the script's directory
dotenv_path = os.path.join(SCRIPT_DIR, '.env')

# Debug: Print paths
print(f"Script directory: {SCRIPT_DIR}")
print(f"Current working directory: {os.getcwd()}")
print(f"Looking for .env at: {dotenv_path}")
print(f".env exists: {os.path.exists(dotenv_path)}")

# Try to load from script directory first, then fall back to current directory
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(f"Loaded .env from: {dotenv_path}")
else:
    # Try current working directory
    cwd_dotenv = os.path.join(os.getcwd(), '.env')
    if os.path.exists(cwd_dotenv):
        load_dotenv(cwd_dotenv)
        print(f"Loaded .env from: {cwd_dotenv}")
    else:
        load_dotenv()  # Let dotenv search default locations
        print("Attempted to load .env from default locations")

# Setup Mistral API
MISTRAL_MODEL = "mistral-small-latest" 
api_key = os.getenv("MISTRAL_API_KEY", "")

# Debug: Check if key was loaded (show only first/last 4 chars for security)
if api_key:
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
    print(f"API Key loaded: {masked_key} (length: {len(api_key)})")
else:
    print("API Key NOT loaded from environment")

API_CONFIGURED = False

if api_key:
    API_CONFIGURED = True
else:
    print("ALERT: MISTRAL_API_KEY is not loaded. API calls will be skipped.")

# Setup local folder (use absolute path to avoid confusion)
LOCAL_SAVE_DIR = os.path.join(os.getcwd(), 'spreads-yo')
os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)
print(f"Local save directory: {LOCAL_SAVE_DIR}")

def quick_api_test():
    """
    Performs a simple, quick connectivity test using the official Mistral SDK.
    """
    global API_CONFIGURED

    if not API_CONFIGURED:
        print("ALERT: MISTRAL_API_KEY is not configured. Skipping connectivity test.")
        return

    try:
        print("--- Testing Mistral API connectivity (Quick Check) ---")
        
        # Using the SDK with context manager as requested
        with Mistral(api_key=api_key) as mistral:
            res = mistral.chat.complete(
                model=MISTRAL_MODEL,
                messages=[
                    {"role": "user", "content": "Generate the single word: Connected"}
                ],
                # 15 Relevant Parameters implementation (subset for quick test)
                stream=False,
                temperature=0.1,
                max_tokens=10,
                top_p=1.0,
                random_seed=42,
                safe_prompt=False
            )

        if res and res.choices and res.choices[0].message.content:
            generated_text = res.choices[0].message.content.strip()
            if 'connected' in generated_text.lower():
                print(f"SUCCESS: Mistral API is operational using {MISTRAL_MODEL}. Test response: '{generated_text}'")
            else:
                print(f"WARNING: Mistral API connected but returned unexpected text: '{generated_text}'")
        else:
             print("WARNING: Mistral API returned empty response structure.")
            
    except Exception as e:
        print(f"ERROR: Failed to call Mistral API. Details: {e}")
        API_CONFIGURED = False # Stop further AI calls if the test fails.

def calculate_technical_indicators(df):
    """Calculates 12 technical and market-based indicators."""
    
    # 1. 50-day Simple Moving Average (SMA 50)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    # 2. 200-day Simple Moving Average (SMA 200)
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # 3. Relative Strength Index (RSI 14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. Moving Average Convergence Divergence (MACD)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 5. Bollinger Band %B
    window = 20
    std = df['Close'].rolling(window=window).std()
    ma = df['Close'].rolling(window=window).mean()
    df['Upper_Band'] = ma + (std * 2)
    df['Lower_Band'] = ma - (std * 2)
    df['BB_PercentB'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])
    
    # 6. Historical Volatility (30-day HV)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['HV_30d'] = df['Log_Return'].rolling(window=30).std() * np.sqrt(252) 
    
    # 7. Average True Range (ATR 14)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(span=14, adjust=False).mean()

    # 8. Average Volume (50-day)
    df['Avg_Volume_50'] = df['Volume'].rolling(window=50).mean()
    
    # 9. Rate of Change (ROC 10-day)
    df['ROC_10d'] = df['Close'].diff(periods=10) / df['Close'].shift(10) * 100

    # 10. Stochastic Oscillator %K
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    
    # 11. Money Flow Index (MFI 14)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    money_ratio = positive_flow / negative_flow
    df['MFI'] = 100 - (100 / (1 + money_ratio))

    # 12. Accumulation/Distribution Line (A/D Line)
    money_flow_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    money_flow_volume = money_flow_multiplier * df['Volume']
    df['ADL'] = money_flow_volume.cumsum()
    
    df.dropna(inplace=True)
    latest_indicators = df.iloc[-1]
    
    indicators_to_return = {
        'Current_Price': latest_indicators['Close'],
        'SMA_50': latest_indicators['SMA_50'],
        'SMA_200': latest_indicators['SMA_200'],
        'RSI': latest_indicators['RSI'],
        'MACD_Value': latest_indicators['MACD'],
        'MACD_Signal': latest_indicators['Signal_Line'],
        'MACD_Bullish': latest_indicators['MACD'] > latest_indicators['Signal_Line'],
        'BB_PercentB': latest_indicators['BB_PercentB'],
        'HV_30d': latest_indicators['HV_30d'],
        'ATR': latest_indicators['ATR'],
        'Avg_Volume_50': latest_indicators['Avg_Volume_50'],
        'ROC_10d': latest_indicators['ROC_10d'],
        'Stoch_K': latest_indicators['Stoch_K'],
        'MFI': latest_indicators['MFI'],
        'ADL': latest_indicators['ADL']
    }
    return indicators_to_return

def get_safe_value(row, col_name):
    """Safely gets a value from a row if the column exists, else returns None."""
    if col_name in row.index:
        return row[col_name]
    return None

def select_spread_strikes(chain, current_price, spread_type, expiration):
    """
    Selects hypothetical strikes for a credit spread.
    Handles missing Greek columns gracefully.
    """
    # Define desired columns, checking what actually exists in the chain
    base_cols = ['strike', 'lastPrice', 'bid', 'ask', 'contractSymbol', 'side']
    greek_cols = ['delta', 'theta', 'vega']
    available_cols = [c for c in greek_cols if c in chain.columns]
    cols_to_use = base_cols + available_cols
    
    # Filter chain to relevant columns
    chain = chain[cols_to_use].copy()
    
    if spread_type == 'PUT_CREDIT':
        puts = chain[chain['side'] == 'put']
        otm_puts = puts[puts['strike'] < current_price]
        if otm_puts.empty: return None
        
        target = current_price * 0.97
        sell_row = otm_puts.iloc[(otm_puts['strike'] - target).abs().argsort()[:1]]
        if sell_row.empty: return None
        sell_strike = sell_row.iloc[0]
        
        buy_cand = puts[puts['strike'] < sell_strike['strike']]
        if buy_cand.empty: return None
        buy_row = buy_cand.iloc[(buy_cand['strike'] - (sell_strike['strike'] - 5)).abs().argsort()[:1]]
        buy_strike = buy_row.iloc[0]
        
        mid_sell = (sell_strike['bid'] + sell_strike['ask']) / 2
        mid_buy = (buy_strike['bid'] + buy_strike['ask']) / 2
        premium = mid_sell - mid_buy
        
        return {
            'type': 'Put Credit Spread',
            'expiration': expiration,
            'sell_strike': sell_strike['strike'],
            'buy_strike': buy_strike['strike'],
            'mid_premium': premium,
            'max_profit': round(premium * 100, 2),
            'max_loss': round((abs(sell_strike['strike'] - buy_strike['strike']) * 100) - (premium * 100), 2),
            'delta_sell': get_safe_value(sell_strike, 'delta'),
            'theta_sell': get_safe_value(sell_strike, 'theta'),
            'vega_sell': get_safe_value(sell_strike, 'vega'),
        }
    
    elif spread_type == 'CALL_CREDIT':
        calls = chain[chain['side'] == 'call']
        otm_calls = calls[calls['strike'] > current_price]
        if otm_calls.empty: return None
        
        target = current_price * 1.03
        sell_row = otm_calls.iloc[(otm_calls['strike'] - target).abs().argsort()[:1]]
        if sell_row.empty: return None
        sell_strike = sell_row.iloc[0]
        
        buy_cand = calls[calls['strike'] > sell_strike['strike']]
        if buy_cand.empty: return None
        buy_row = buy_cand.iloc[(buy_cand['strike'] - (sell_strike['strike'] + 5)).abs().argsort()[:1]]
        buy_strike = buy_row.iloc[0]
        
        mid_sell = (sell_strike['bid'] + sell_strike['ask']) / 2
        mid_buy = (buy_strike['bid'] + buy_strike['ask']) / 2
        premium = mid_sell - mid_buy
        
        return {
            'type': 'Call Credit Spread',
            'expiration': expiration,
            'sell_strike': sell_strike['strike'],
            'buy_strike': buy_strike['strike'],
            'mid_premium': premium,
            'max_profit': round(premium * 100, 2),
            'max_loss': round((abs(sell_strike['strike'] - buy_strike['strike']) * 100) - (premium * 100), 2),
            'delta_sell': get_safe_value(sell_strike, 'delta'),
            'theta_sell': get_safe_value(sell_strike, 'theta'),
            'vega_sell': get_safe_value(sell_strike, 'vega'),
        }
    return None

def generate_ai_rationale(spread_data, indicators):
    """Uses Mistral SDK to generate a trading rationale with exponential backoff"""
    global API_CONFIGURED
    
    if not API_CONFIGURED:
        return "AI rationale skipped: Mistral API is not configured or failed the connection test."
    
    max_retries = 3
    initial_delay = 2  # seconds
    
    prompt = f"""
    You are a quantitative financial analyst. Generate a detailed, hypothetical analysis paragraph for a {spread_data['type']} on {TICKER}.
    
    **Market Context (Technical Indicators):**
    - Current Price: ${indicators['Current_Price']:.2f}
    - Trend (SMA 50/200): SMA50={indicators['SMA_50']:.2f}, SMA200={indicators['SMA_200']:.2f}
    - Momentum (RSI): {indicators['RSI']:.2f} (Overbought>70, Oversold<30)
    - MACD: Value={indicators['MACD_Value']:.2f}, Signal={indicators['MACD_Signal']:.2f} ({'Bullish' if indicators['MACD_Bullish'] else 'Bearish'})
    - Stochastic %K: {indicators['Stoch_K']:.2f}
    - Volatility (HV 30d): {indicators['HV_30d']:.2f}
    - Bollinger Band %B: {indicators['BB_PercentB']:.2f}
    - Money Flow Index: {indicators['MFI']:.2f}
    - Rate of Change (10d): {indicators['ROC_10d']:.2f}%
    
    **Trade Structure:**
    - Expiration: {spread_data['expiration']}
    - Sell Strike: ${spread_data['sell_strike']:.2f}
    - Buy Strike: ${spread_data['buy_strike']:.2f}
    - Est. Max Profit: ${spread_data['max_profit']}
    - Est. Max Risk: ${spread_data['max_loss']}
    - Greeks (Short Leg): Delta={spread_data.get('delta_sell', 'N/A')}, Theta={spread_data.get('theta_sell', 'N/A')}

    **Instructions:**
    - Write a sophisticated, text-heavy paragraph analyzing why this specific setup might be interesting given the indicators. 
    - Connect the specific indicator values (e.g., "With RSI at X...") to the trade thesis.
    - If the trade is a Put Credit Spread, look for bullish/neutral signs. If Call Credit Spread, look for bearish/neutral signs.
    - Mention the risk/reward profile.
    - DISCLAIMER: This is hypothetical analysis, not financial advice.
    """
    
    for attempt in range(max_retries):
        try:
            # Implementing the 15 most relevant parameters
            with Mistral(api_key=api_key) as mistral:
                res = mistral.chat.complete(
                    # 1. Model (Required)
                    model=MISTRAL_MODEL,
                    # 2. Messages (Required)
                    messages=[
                        {"role": "system", "content": "You are a quantitative financial analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    # 3. Temperature: Balance between creativity and determinism (0.5)
                    temperature=0.5,
                    # 4. Max Tokens: Ensure enough space for the paragraph (800)
                    max_tokens=800,
                    # 5. Top P: Nucleus sampling (1.0 = consider all valid tokens)
                    top_p=1.0,
                    # 6. Frequency Penalty: Discourage exact repetition of words (0.1)
                    frequency_penalty=0.1,
                    # 7. Presence Penalty: Encourage new topics/variety (0.1)
                    presence_penalty=0.1,
                    # 8. Stream: False (we want the full response at once)
                    stream=False,
                    # 9. Random Seed: For deterministic results if needed (42)
                    random_seed=42,
                    # 10. Safe Prompt: Explicitly set safety check
                    safe_prompt=False,
                    # 11. N: Number of completions (1)
                    n=1,
                    # 12. Response Format: Explicitly ask for text
                    response_format={"type": "text"},
                    # 13. Tool Choice: Force 'none' to prevent tool calling logic
                    tool_choice="none",
                    # 14. Tools: Explicitly null to avoid confusion
                    tools=None,
                    # 15. Stop: None (let it finish naturally)
                    stop=None
                )
            
            if res and res.choices and res.choices[0].message.content:
                return res.choices[0].message.content.strip()
            else:
                raise ValueError("Empty response received from Mistral API")
        
        except Exception as e:
            error_message = str(e)
            
            # Check for rate limit errors in the SDK exception
            if "429" in error_message or "Resource has been exhausted" in error_message:
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)  # Exponential backoff
                    print(f"RATE LIMIT ERROR. Retrying in {delay:.1f} seconds (Attempt {attempt + 2}/{max_retries})...")
                    time.sleep(delay)
                else:
                    print(f"FINAL ATTEMPT FAILED for AI rationale: {error_message}")
                    return f"ERROR: Failed after {max_retries} retries. Details: {error_message}"
            else:
                print(f"UNRECOVERABLE ERROR: {error_message}")
                return f"API ERROR: {error_message}"
    
    return "ERROR: Failed to generate AI rationale."

def save_locally(data, filename):
    """Save analysis files locally"""
    print(f"\n💾 Saving locally to: {LOCAL_SAVE_DIR}")
    
    try:
        # Save JSON
        json_filename = filename.replace('.md', '.json')
        json_path = os.path.join(LOCAL_SAVE_DIR, json_filename)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"✅ Saved: {json_filename}")
        
        # Save Markdown
        md_path = os.path.join(LOCAL_SAVE_DIR, filename)
        with open(md_path, 'w') as f:
            f.write(data['markdown_report'])
        print(f"✅ Saved: {filename}")
        
        return True
    except Exception as e:
        print(f"❌ Local save error: {e}")
        return False

def upload_to_gcp(data, filename):
    """Upload analysis to GCP bucket"""
    print(f"\n☁️  Uploading to GCP: {GCP_BUCKET}/options...")
    
    try:
        client = storage.Client()
        bucket = client.bucket(GCP_BUCKET)
        
        date_folder = datetime.now().strftime('%Y-%m-%d')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prefix = f'options/{date_folder}/{TICKER}'
        
        # Upload JSON
        json_filename = filename.replace('.md', '.json')
        blob = bucket.blob(f'{prefix}_{timestamp}.json')
        blob.upload_from_string(json.dumps(data, indent=2, default=str), content_type='application/json')
        print(f"✅ Uploaded: JSON")
        
        # Upload Markdown
        blob = bucket.blob(f'{prefix}_{timestamp}.md')
        blob.upload_from_string(data['markdown_report'], content_type='text/markdown')
        print(f"✅ Uploaded: Markdown")
        
        print(f"\n✅ Files uploaded to gs://{GCP_BUCKET}/options/{date_folder}/")
        return True
    except Exception as e:
        print(f"❌ GCP upload error: {e}")
        return False

def main_analysis():
    print(f"--- AI-Enhanced Options Analysis: {TICKER} ---")
    
    # Test API
    quick_api_test()
    
    # Fetch data
    ticker = yf.Ticker(TICKER)
    df = ticker.history(period="max")
    
    if df.empty:
        print("ERROR: No historical data")
        return
    
    # Calculate indicators
    indicators = calculate_technical_indicators(df.copy())
    print(f"Latest Price: ${indicators['Current_Price']:.2f}")
    
    # Get option expirations
    try:
        available_exp = ticker.options
    except Exception:
        print("ERROR: Could not fetch options")
        return
    
    if not available_exp:
        print("ERROR: No options available")
        return
    
    # Select 6 expirations
    target_days = [14, 30, 45, 60, 90, 180]
    selected_exp = []
    for t in target_days:
        target_date = date.today() + timedelta(days=t)
        closest = min(available_exp, key=lambda x: abs((date.fromisoformat(x) - target_date).days))
        if closest not in selected_exp:
            selected_exp.append(closest)
    
    selected_exp = selected_exp[:6]
    
    # Analyze spreads
    all_spreads = []
    api_delay = 3.0
    
    for exp in selected_exp:
        print(f"Processing: {exp}...")
        try:
            chain = ticker.option_chain(exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls['side'] = 'call'
            puts['side'] = 'put'
            
            full_chain = pd.concat([calls, puts], ignore_index=True, sort=False)
            
            # Put credit spread
            pcs = select_spread_strikes(full_chain, indicators['Current_Price'], 'PUT_CREDIT', exp)
            if pcs:
                time.sleep(api_delay)
                pcs['rationale'] = generate_ai_rationale(pcs, indicators)
                all_spreads.append(pcs)
            
            # Call credit spread
            ccs = select_spread_strikes(full_chain, indicators['Current_Price'], 'CALL_CREDIT', exp)
            if ccs:
                time.sleep(api_delay)
                ccs['rationale'] = generate_ai_rationale(ccs, indicators)
                all_spreads.append(ccs)
        
        except Exception as e:
            print(f"  Skipping {exp}: {e}")
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{TICKER}_spread_analysis_{timestamp}.md"
    
    # Build markdown
    md_report = f"# AI-Enhanced Credit Spread Analysis: {TICKER}\n\n"
    md_report += f"**Date:** {date.today()}\n"
    md_report += f"**Reference Price:** ${indicators['Current_Price']:.2f}\n\n"
    
    md_report += "## Technical Landscape (12 Indicators)\n"
    for k, v in indicators.items():
        if isinstance(v, float):
            md_report += f"- **{k}:** {v:.2f}\n"
        else:
            md_report += f"- **{k}:** {v}\n"
    md_report += "\n---\n\n"
    
    for spread in all_spreads:
        md_report += f"### {spread['type']} ({spread['expiration']})\n"
        md_report += f"**Strategy:** Sell ${spread['sell_strike']} / Buy ${spread['buy_strike']}\n"
        md_report += f"**Est. Credit:** ${spread['max_profit']} | **Max Risk:** ${spread['max_loss']}\n"
        md_report += f"**Analysis:**\n{spread['rationale']}\n\n"
        md_report += "---\n"
    
    # Prepare complete data structure
    complete_data = {
        'symbol': TICKER,
        'timestamp': timestamp,
        'date': str(date.today()),
        'indicators': indicators,
        'spreads': all_spreads,
        'markdown_report': md_report
    }
    
    # Save locally
    save_locally(complete_data, filename)
    
    # Upload to GCP
    try:
        upload_to_gcp(complete_data, filename)
    except Exception as e:
        print(f"⚠️  GCP upload skipped: {e}")
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main_analysis()