import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import os
import time
import json
import tempfile
import shutil
from dotenv import load_dotenv
from mistralai import Mistral 

# --- CONFIGURATION ---
TICKER = "GLD" 
DAYS_OF_HISTORY = 90

# 1. Load environment variables from a .env file if one exists
load_dotenv()

# Setup Mistral API
MISTRAL_MODEL = "mistral-small-latest" 
api_key = os.getenv("MISTRAL_API_KEY", "")
API_CONFIGURED = False

if api_key:
    API_CONFIGURED = True
else:
    print("ALERT: MISTRAL_API_KEY is not loaded. API calls will be skipped.")

# --- JSON Encoder Class for NumPy Types ---
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- Atomic File Write Helper ---
def write_file_atomically(filepath, content, is_json=False, cls=None):
    """
    Write content to a file atomically by using a temporary file and then renaming.
    This ensures that if the write is interrupted, the original file is not corrupted.
    
    Args:
        filepath: Destination file path
        content: Content to write (string or object to JSON serialize)
        is_json: If True, content will be JSON serialized
        cls: JSON encoder class (e.g., NumpyEncoder)
    """
    # Create temp file in the same directory as the target to ensure same filesystem
    target_dir = os.path.dirname(filepath)
    if not target_dir:
        target_dir = '.'
    
    try:
        # Write to temporary file
        with tempfile.NamedTemporaryFile(
            mode='w' if not is_json else 'w',
            dir=target_dir,
            delete=False,
            suffix='.tmp'
        ) as tmp_file:
            tmp_path = tmp_file.name
            
            if is_json:
                json.dump(content, tmp_file, indent=4, cls=cls)
            else:
                tmp_file.write(content)
        
        # Atomically rename temp file to final destination
        # On Unix, rename is atomic. On Windows, we need to remove the target first.
        if os.path.exists(filepath):
            os.remove(filepath)
        
        shutil.move(tmp_path, filepath)
        
    except Exception as e:
        # Clean up temp file if something goes wrong
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
        raise Exception(f"Failed to write file {filepath}: {e}")

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
        
        with Mistral(api_key=api_key) as mistral:
            res = mistral.chat.complete(
                model=MISTRAL_MODEL,
                messages=[
                    {"role": "user", "content": "Generate the single word: Connected"}
                ],
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
        API_CONFIGURED = False 


# --- Technical Indicator Calculation Functions ---

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
    base_cols = ['strike', 'lastPrice', 'bid', 'ask', 'contractSymbol', 'side']
    greek_cols = ['delta', 'theta', 'vega']
    available_cols = [c for c in greek_cols if c in chain.columns]
    cols_to_use = base_cols + available_cols
    
    chain = chain[cols_to_use].copy()

    if spread_type == 'PUT_CREDIT':
        # Bullish: Sell OTM Put (Strike < Price)
        puts = chain[chain['side'] == 'put']
        otm_puts = puts[puts['strike'] < current_price]
        
        if otm_puts.empty: return None
        
        target_strike = current_price * 0.97
        sell_row = otm_puts.iloc[(otm_puts['strike'] - target_strike).abs().argsort()[:1]]
        if sell_row.empty: return None
        sell_strike = sell_row.iloc[0]
        
        buy_candidates = puts[puts['strike'] < sell_strike['strike']]
        if buy_candidates.empty: return None
        
        buy_row = buy_candidates.iloc[(buy_candidates['strike'] - (sell_strike['strike'] - 5)).abs().argsort()[:1]]
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
        # Bearish: Sell OTM Call (Strike > Price)
        calls = chain[chain['side'] == 'call']
        otm_calls = calls[calls['strike'] > current_price]
        
        if otm_calls.empty: return None
        
        target_strike = current_price * 1.03
        sell_row = otm_calls.iloc[(otm_calls['strike'] - target_strike).abs().argsort()[:1]]
        if sell_row.empty: return None
        sell_strike = sell_row.iloc[0]

        buy_candidates = calls[calls['strike'] > sell_strike['strike']]
        if buy_candidates.empty: return None
        
        buy_row = buy_candidates.iloc[(buy_candidates['strike'] - (sell_strike['strike'] + 5)).abs().argsort()[:1]]
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
    """Uses Mistral SDK to generate a trading rationale."""
    global API_CONFIGURED
    
    if not API_CONFIGURED:
        return "AI rationale skipped: Mistral API is not configured or failed the connection test."

    max_retries = 3
    initial_delay = 2  
    
    prompt = f"""
    You are a quantitative financial analyst. Generate a detailed, hypothetical analysis paragraph for a {spread_data['type']} on {TICKER}.
    
    **Market Context (Technical Indicators):**
    - Current Price: ${indicators['Current_Price']:.2f}
    - Trend (SMA 50/200): SMA50={indicators['SMA_50']:.2f}, SMA200={indicators['SMA_200']:.2f}
    - Momentum (RSI): {indicators['RSI']:.2f}
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
    
    **Instructions:**
    - Write a sophisticated, text-heavy paragraph analyzing why this specific setup might be interesting given the indicators. 
    - DISCLAIMER: This is hypothetical analysis, not financial advice.
    """
    
    for attempt in range(max_retries):
        try:
            with Mistral(api_key=api_key) as mistral:
                res = mistral.chat.complete(
                    model=MISTRAL_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a quantitative financial analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=800,
                    top_p=1.0,
                    frequency_penalty=0.1,
                    presence_penalty=0.1,
                    stream=False,
                    random_seed=42,
                    safe_prompt=False,
                    n=1,
                    response_format={"type": "text"},
                    tool_choice="none",
                    tools=None,
                    stop=None
                )

            if res and res.choices and res.choices[0].message.content:
                return res.choices[0].message.content.strip()
            else:
                raise ValueError("Empty response received from Mistral API")
        
        except Exception as e:
            error_message = str(e)
            if "429" in error_message or "Resource has been exhausted" in error_message:
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)  
                    print(f"RATE LIMIT ERROR. Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    return f"ERROR: Failed after {max_retries} retries. Details: {error_message}"
            else:
                return f"API ERROR: {error_message}"
    
    return "ERROR: Failed to generate AI rationale."

def main_analysis():
    print(f"--- Running AI-Enhanced Technical Analysis on {TICKER} ---")
    
    # 0. Test API Connection
    quick_api_test()

    # 1. Fetch Data
    ticker_yf = yf.Ticker(TICKER)
    df = ticker_yf.history(period="max")
    
    if df.empty:
        print("Error: No historical data found.")
        return

    # 2. Calculate Indicators
    indicators = calculate_technical_indicators(df.copy())
    print(f"Latest Price: ${indicators['Current_Price']:.2f}")

    # 3. Get Expirations
    try:
        available_expirations = ticker_yf.options
    except Exception:
        print("Error: Could not fetch option expirations.")
        return

    if not available_expirations:
        print("Error: No options data available.")
        return

    # Select 6 expirations spread out over time
    target_days = [14, 30, 45, 60, 90, 180] 
    selected_expirations = []
    for t in target_days:
        target_date = date.today() + timedelta(days=t)
        closest = min(available_expirations, key=lambda x: abs((date.fromisoformat(x) - target_date).days))
        if closest not in selected_expirations:
            selected_expirations.append(closest)
    
    selected_expirations = selected_expirations[:6]

    # 4. Analyze Spreads
    all_spread_analysis = []
    API_CALL_DELAY = 3.0 
    
    for expiration in selected_expirations:
        print(f"Processing expiration: {expiration}...")
        try:
            chain = ticker_yf.option_chain(expiration)
            calls = chain.calls
            puts = chain.puts
            calls['side'] = 'call'
            puts['side'] = 'put'
            
            full_chain = pd.concat([calls, puts], ignore_index=True, sort=False)
            
            pcs = select_spread_strikes(full_chain, indicators['Current_Price'], 'PUT_CREDIT', expiration)
            if pcs:
                time.sleep(API_CALL_DELAY) 
                pcs['rationale'] = generate_ai_rationale(pcs, indicators)
                all_spread_analysis.append(pcs)
                
            ccs = select_spread_strikes(full_chain, indicators['Current_Price'], 'CALL_CREDIT', expiration)
            if ccs:
                time.sleep(API_CALL_DELAY)
                ccs['rationale'] = generate_ai_rationale(ccs, indicators)
                all_spread_analysis.append(ccs)
                
        except Exception as e:
            print(f"  Skipping {expiration}: {e}")

    # 5. Write Report (Markdown and JSON) with atomic writes
    output_folder = "spreads-yo"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Generate unique filename based on symbol and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_filename = os.path.join(output_folder, f"{TICKER}_spread_analysis_{timestamp}.md")

    # Build Markdown content
    markdown_content = f"# AI-Enhanced Credit Spread Analysis: {TICKER}\n\n"
    markdown_content += f"**Date:** {date.today()}\n"
    markdown_content += f"**Reference Price:** ${indicators['Current_Price']:.2f}\n\n"
    
    markdown_content += "## Technical Landscape (12 Indicators)\n"
    for k, v in indicators.items():
        if isinstance(v, float):
            markdown_content += f"- **{k}:** {v:.2f}\n"
        else:
            markdown_content += f"- **{k}:** {v}\n"
    markdown_content += "\n---\n\n"
    
    for spread in all_spread_analysis:
        markdown_content += f"### {spread['type']} ({spread['expiration']})\n"
        markdown_content += f"**Strategy:** Sell ${spread['sell_strike']} / Buy ${spread['buy_strike']}\n"
        markdown_content += f"**Est. Credit:** ${spread['max_profit']} | **Max Risk:** ${spread['max_loss']}\n"
        markdown_content += f"**Analysis:**\n{spread['rationale']}\n\n"
        markdown_content += "---\n"
    
    # Write Markdown atomically
    try:
        write_file_atomically(md_filename, markdown_content)
        print(f"Markdown report saved to '{md_filename}'.")
    except Exception as e:
        print(f"ERROR writing Markdown: {e}")

    # Save Raw List JSON (original request) atomically
    json_list_filename = os.path.join(output_folder, f"{TICKER}_spread_analysis_{timestamp}.json")
    try:
        write_file_atomically(json_list_filename, all_spread_analysis, is_json=True, cls=NumpyEncoder)
        print(f"Raw List JSON saved to '{json_list_filename}'.")
    except Exception as e:
        print(f"ERROR writing JSON list: {e}")

    # --- FINAL STEP: Generate Structured JSON for Next.js with atomic write ---
    output_folder_json = "spreads-yo-json"
    if not os.path.exists(output_folder_json):
        os.makedirs(output_folder_json)

    final_json_filename = os.path.join(output_folder_json, f"{TICKER}_analysis_{timestamp}.json")

    # Build the structured data object
    report_data = {
        "ticker": TICKER,
        "date": str(date.today()),
        "reference_price": indicators.get('Current_Price'),
        "technical_landscape": indicators,  
        "spreads": []
    }

    # Cleanly map the spreads into the JSON structure
    for spread in all_spread_analysis:
        report_data["spreads"].append({
            "type": spread['type'],
            "expiration": spread['expiration'],
            "strategy": {
                "sell": spread['sell_strike'],
                "buy": spread['buy_strike']
            },
            "financials": {
                "est_credit": spread['max_profit'],
                "max_risk": spread['max_loss']
            },
            "analysis": spread['rationale']
        })

    # Write structured JSON atomically
    try:
        write_file_atomically(final_json_filename, report_data, is_json=True, cls=NumpyEncoder)
        print(f"Done. Next.js ready JSON saved to '{final_json_filename}'.")
    except Exception as e:
        print(f"ERROR writing structured JSON: {e}")

    print(f"\n✅ All files saved successfully using atomic writes.")

if __name__ == "__main__":
    main_analysis()