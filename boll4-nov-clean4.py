#!/usr/bin/env python3
"""
Enhanced AI-Enhanced Options Spread Analysis with GCP Storage
- Uses pathlib, robust logging, retries & exponential backoff
- Prefers google.cloud.storage client but falls back to gcloud CLI (subprocess) if needed
- Safer handling of secrets; supports Application Default Credentials (recommended)
- Improved Mistral SDK use with retry/backoff
- More robust technical indicator calculations (safer divisions, NaN handling)
"""

from __future__ import annotations
import argparse
import json
import logging
import math
import subprocess
import sys
import time

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# External cloud & AI SDKs (ensure installed + configured)
from google.cloud import storage
from mistralai import Mistral

# ---------- Configuration ----------
TICKER = "SPY"
DAYS_OF_HISTORY = 90  # use a longer window for indicators calculation; change as desired
GCP_BUCKET = "ttb-bucket1"
LOCAL_SAVE_DIR = Path.cwd() / "spreads-yo"
LOCAL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Mistral config
MISTRAL_MODEL = "mistral-small-latest"  # keep as before
# load .env if exists
load_dotenv()  # will read .env in cwd or environment

# Read API keys from environment (prefer ADC for GCP, avoid injecting service account keys)
MISTRAL_API_KEY = None
if "MISTRAL_API_KEY" in (env := dict()):
    pass  # keep the pattern - we will read from os.environ below

import os
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
# Note: For GCP, prefer Application Default Credentials (ADC), or configure `gcloud auth application-default login`
# If you must use a service account key, use environment variable GOOGLE_APPLICATION_CREDENTIALS pointing to the key file
# but avoid putting long keys into environment variables.

# ---------- Logging ----------
LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("spread_analysis")

# ---------- Utilities ----------
def safe_div(numerator: float, denominator: float, default: Optional[float] = None) -> Optional[float]:
    try:
        if denominator is None or math.isclose(denominator, 0.0):
            return default
        return numerator / denominator
    except Exception:
        return default

def retry_with_backoff(
    func,
    max_retries: int = 4,
    initial_delay: float = 1.0,
    multiplier: float = 2.0,
    retry_on_exceptions: tuple = (Exception,),
    on_retry: Optional[callable] = None,
    **kwargs,
):
    """
    Generic retry wrapper. `func` is called with kwargs.
    """
    attempt = 0
    while True:
        try:
            return func(**kwargs)
        except retry_on_exceptions as e:
            attempt += 1
            if attempt > max_retries:
                logger.exception("Max retries exceeded for function %s", getattr(func, "__name__", str(func)))
                raise
            delay = initial_delay * (multiplier ** (attempt - 1))
            logger.warning("Retryable error: %s. Retrying in %.1f seconds (attempt %d/%d).", e, delay, attempt, max_retries)
            if on_retry:
                try:
                    on_retry(attempt, e)
                except Exception:
                    pass
            time.sleep(delay)

# ---------- Technical indicators ----------
def calculate_technical_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Input: df must contain ['Open','High','Low','Close','Volume'] indexed by date
    Returns: dictionary of the latest computed indicators (dropna handled internally)
    """
    df = df.copy().sort_index()

    # Ensure numeric and drop rows with missing price/volume
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    df = df.dropna(subset=required_columns)
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 50 and 200 SMA
    df["SMA_50"] = df["Close"].rolling(window=50, min_periods=10).mean()
    df["SMA_200"] = df["Close"].rolling(window=200, min_periods=50).mean()

    # RSI (14) using simple method
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=14, min_periods=7).mean()
    roll_down = down.rolling(window=14, min_periods=7).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Bullish"] = df["MACD"] > df["Signal_Line"]

    # Bollinger Bands %B
    window = 20
    ma = df["Close"].rolling(window=window, min_periods=10).mean()
    std = df["Close"].rolling(window=window, min_periods=10).std()
    df["Upper_Band"] = ma + 2 * std
    df["Lower_Band"] = ma - 2 * std
    df["BB_PercentB"] = (df["Close"] - df["Lower_Band"]) / (df["Upper_Band"] - df["Lower_Band"]).replace(0, np.nan)

    # Historical Volatility (30-day)
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["HV_30d"] = df["Log_Return"].rolling(window=30, min_periods=10).std() * np.sqrt(252)

    # ATR (14)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=14, adjust=False).mean()

    # Avg volume (50)
    df["Avg_Volume_50"] = df["Volume"].rolling(window=50, min_periods=10).mean()

    # ROC 10d
    df["ROC_10d"] = (df["Close"].diff(periods=10) / df["Close"].shift(10)) * 100

    # Stochastic %K
    low_14 = df["Low"].rolling(window=14, min_periods=7).min()
    high_14 = df["High"].rolling(window=14, min_periods=7).max()
    df["Stoch_K"] = 100 * safe_div((df["Close"] - low_14), (high_14 - low_14), default=np.nan)

    # Money Flow Index (14)
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = typical_price * df["Volume"]
    positive_flow = mf.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
    negative_flow = mf.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    money_ratio = safe_div(positive_flow, negative_flow, default=np.nan)
    df["MFI"] = 100 - (100 / (1 + money_ratio))

    # Accumulation/Distribution Line (ADL)
    denom = (df["High"] - df["Low"]).replace(0, np.nan)
    money_flow_multiplier = safe_div(((df["Close"] - df["Low"]) - (df["High"] - df["Close"])), denom, default=0.0)
    df["Money_Flow_Volume"] = money_flow_multiplier * df["Volume"]
    df["ADL"] = df["Money_Flow_Volume"].cumsum()

    df = df.dropna(how="all")  # drop rows entirely NaN
    if df.empty:
        raise ValueError("Dataframe empty after indicator calculations")

    latest = df.iloc[-1].to_dict()

    # Build indicator dict with safe defaults
    indicators = {
        "Current_Price": float(latest.get("Close", np.nan)),
        "SMA_50": float(latest.get("SMA_50", np.nan)),
        "SMA_200": float(latest.get("SMA_200", np.nan)),
        "RSI": float(latest.get("RSI", np.nan)),
        "MACD_Value": float(latest.get("MACD", np.nan)),
        "MACD_Signal": float(latest.get("Signal_Line", np.nan)),
        "MACD_Bullish": bool(latest.get("MACD_Bullish", False)),
        "BB_PercentB": float(latest.get("BB_PercentB", np.nan)),
        "HV_30d": float(latest.get("HV_30d", np.nan)),
        "ATR": float(latest.get("ATR", np.nan)),
        "Avg_Volume_50": float(latest.get("Avg_Volume_50", np.nan)),
        "ROC_10d": float(latest.get("ROC_10d", np.nan)),
        "Stoch_K": float(latest.get("Stoch_K", np.nan)),
        "MFI": float(latest.get("MFI", np.nan)),
        "ADL": float(latest.get("ADL", np.nan)),
    }
    return indicators

# ---------- Mistral integration ----------
def quick_mistral_test(api_key: str, model: str) -> bool:
    if not api_key:
        logger.warning("Mistral API key not configured. Skipping Mistral test.")
        return False

    try:
        logger.info("Testing Mistral connectivity...")
        with Mistral(api_key=api_key) as mistral:
            res = mistral.chat.complete(
                model=model,
                messages=[{"role": "user", "content": "Respond with a single word: Connected"}],
                temperature=0.1,
                max_tokens=10,
                stream=False,
            )
        # Validate response
        if res and getattr(res, "choices", None):
            text = getattr(res.choices[0].message, "content", "")
            if "connected" in text.lower():
                logger.info("Mistral connectivity OK (model=%s).", model)
                return True
            logger.warning("Mistral returned unexpected text: %s", text)
            return True
        logger.warning("Mistral returned no usable content.")
        return False
    except Exception as e:
        logger.exception("Mistral test failed: %s", e)
        return False

def generate_ai_rationale(
    api_key: str,
    model: str,
    spread_data: Dict[str, Any],
    indicators: Dict[str, Any],
    max_retries: int = 3,
) -> str:
    """
    Generate a paragraph explaining the trade using Mistral. Uses exponential backoff on failures.
    Returns a string (or error text) — never raises.
    """
    if not api_key:
        return "AI rationale skipped: Mistral API key not configured."

    # Build prompt (careful with types)
    prompt = f"""
You are a quantitative financial analyst. Generate a single detailed paragraph analyzing a {spread_data['type']} for {TICKER}.

Market context:
- Current Price: ${indicators['Current_Price']:.2f}
- SMA50: {indicators['SMA_50']:.2f}, SMA200: {indicators['SMA_200']:.2f}
- RSI: {indicators['RSI']:.2f}
- MACD: {indicators['MACD_Value']:.2f} (Signal {indicators['MACD_Signal']:.2f})
- Stochastic %K: {indicators['Stoch_K']:.2f}
- HV_30d: {indicators['HV_30d']:.2f}
- BB%: {indicators['BB_PercentB']:.2f}
- MFI: {indicators['MFI']:.2f}
Trade specifics:
- Sell {spread_data['sell_strike']}, Buy {spread_data['buy_strike']}, Exp: {spread_data['expiration']}
- Est credit: ${spread_data['max_profit']}, Max loss: ${spread_data['max_loss']}
Provide a sophisticated paragraph linking indicators to the trade thesis. Include risk/reward and short disclaimer.
"""

    def call_mistral_once(api_key_local: str, model_local: str, prompt_text: str):
        with Mistral(api_key=api_key_local) as mistral:
            res = mistral.chat.complete(
                model=model_local,
                messages=[{"role": "system", "content": "You are a quantitative financial analyst."},
                          {"role": "user", "content": prompt_text}],
                temperature=0.5,
                max_tokens=600,
                top_p=1.0,
                frequency_penalty=0.1,
                presence_penalty=0.1,
                stream=False,
                n=1,
                response_format={"type": "text"},
            )
        # Try to extract text safely
        try:
            text = res.choices[0].message.content.strip()
            return text
        except Exception:
            # Fallback: attempt attribute-style extraction
            try:
                return str(res)
            except Exception:
                raise ValueError("Empty response structure from Mistral")

    try:
        return retry_with_backoff(
            func=call_mistral_once,
            max_retries=max_retries,
            initial_delay=2.0,
            multiplier=2.0,
            retry_on_exceptions=(Exception,),
            api_key_local=api_key,
            model_local=model,
            prompt_text=prompt,
        )
    except Exception as e:
        logger.exception("Failed to generate AI rationale: %s", e)
        return f"ERROR: Failed to generate AI rationale. Details: {e}"

# ---------- GCP upload helpers ----------
def upload_with_python_client(bucket_name: str, files: List[Path], destination_prefix: str) -> bool:
    """
    Uploads list of local files to bucket using google.cloud.storage.
    Returns True on success, False on any error.
    """
    logger.info("Uploading using google.cloud.storage client to bucket %s", bucket_name)
    try:
        client = storage.Client()  # will use ADC or GOOGLE_APPLICATION_CREDENTIALS
        bucket = client.bucket(bucket_name)
        if not bucket.exists():
            logger.warning("Bucket %s does not appear to exist or is inaccessible.", bucket_name)
            # Still continue: bucket.exists() may require permissions; try upload anyway
        for local_path in files:
            blob_name = f"{destination_prefix}/{local_path.name}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(local_path))
            logger.info("Uploaded %s -> gs://%s/%s", local_path.name, bucket_name, blob_name)
        return True
    except Exception as e:
        logger.exception("Python client upload failed: %s", e)
        return False

def upload_with_gcloud_cli(bucket_name: str, files: List[Path], destination_prefix: str) -> bool:
    """
    Fallback upload using `gcloud storage cp` (gcloud CLI must be installed + authenticated).
    Returns True on success.
    """
    logger.info("Attempting upload via gcloud CLI (fallback).")
    try:
        for local_path in files:
            dest = f"gs://{bucket_name}/{destination_prefix}/{local_path.name}"
            cmd = ["gcloud", "storage", "cp", str(local_path), dest, "--quiet"]
            logger.debug("Running CLI command: %s", " ".join(cmd))
            completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                logger.error("gcloud cp failed for %s: %s %s", local_path, completed.stderr, completed.stdout)
                return False
            logger.info("gcloud cp succeeded for %s -> %s", local_path.name, dest)
        return True
    except FileNotFoundError:
        logger.error("gcloud CLI not found. Install and authenticate gcloud or enable ADC.")
        return False
    except Exception as e:
        logger.exception("gcloud CLI upload failed: %s", e)
        return False

def upload_to_gcp_with_fallback(bucket_name: str, files: List[Path], base_prefix: str) -> bool:
    """
    Try python client, then CLI fallback. Returns True on success.
    """
    date_folder = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = f"{base_prefix}/{date_folder}/{timestamp}"
    # try python client first
    ok = upload_with_python_client(bucket_name, files, prefix)
    if ok:
        return True
    # fallback to gcloud CLI
    ok_cli = upload_with_gcloud_cli(bucket_name, files, prefix)
    return ok_cli

# ---------- Spread selection ----------
def get_safe_value(row: pd.Series, name: str):
    if name in row.index:
        return row[name]
    return None

def select_spread_strikes(chain: pd.DataFrame, current_price: float, spread_type: str, expiration: str) -> Optional[Dict[str, Any]]:
    """
    Minimal, robust selection for a 5-point width credit spread near-target.
    Returns a dict or None.
    """
    if chain.empty:
        return None
    base_cols = ["strike", "lastPrice", "bid", "ask", "contractSymbol", "side"]
    # normalize column names if present
    for col in base_cols:
        if col not in chain.columns:
            chain[col] = np.nan

    chain = chain.copy()
    # Some vendors return floats as ints; ensure numeric
    chain["strike"] = pd.to_numeric(chain["strike"], errors="coerce")

    greek_cols = ["delta", "theta", "vega"]
    available_greeks = [c for c in greek_cols if c in chain.columns]
    cols_to_use = base_cols + available_greeks
    chain = chain[cols_to_use].copy()

    greek_cols = ["delta", "theta", "vega"]
    available_greeks = [c for c in greek_cols if c in chain.columns]

    if spread_type == "PUT_CREDIT":
        puts = chain[chain.get("side", "") == "put"] if "side" in chain.columns else chain[chain.get("contractSymbol", "").str.contains("PUT", na=False)]
        puts = puts.dropna(subset=["strike"])
        otm_puts = puts[puts["strike"] < current_price]
        if otm_puts.empty:
            return None

        # choose a sell strike ~3% below current price (closest)
        target = current_price * 0.97
        sell_row = otm_puts.iloc[(otm_puts["strike"] - target).abs().argsort()[:1]]
        if sell_row.empty:
            return None
        sell = sell_row.iloc[0]
        # choose buy 5 points lower than sell
        buy_candidates = otm_puts[otm_puts["strike"] < sell["strike"]]
        if buy_candidates.empty:
            return None
        desired_buy = sell["strike"] - 5.0
        buy_row = buy_candidates.iloc[(buy_candidates["strike"] - desired_buy).abs().argsort()[:1]]
        if buy_row.empty:
            return None
        buy = buy_row.iloc[0]

        mid_sell = safe_div(sell.get("bid", np.nan) + sell.get("ask", np.nan), 2, default=np.nan)
        mid_buy = safe_div(buy.get("bid", np.nan) + buy.get("ask", np.nan), 2, default=np.nan)
        premium = safe_div(mid_sell - mid_buy, 1, default=np.nan)

        return {
            "type": "Put Credit Spread",
            "expiration": expiration,
            "sell_strike": float(sell["strike"]),
            "buy_strike": float(buy["strike"]),
            "mid_premium": float(premium) if pd.notna(premium) else None,
            "max_profit": round(float(premium * 100) if pd.notna(premium) else 0.0, 2),
            "max_loss": round(float(abs(sell["strike"] - buy["strike"]) * 100 - (premium * 100)) if pd.notna(premium) else 0.0, 2),
            "delta_sell": get_safe_value(sell, "delta"),
            "theta_sell": get_safe_value(sell, "theta"),
            "vega_sell": get_safe_value(sell, "vega"),
        }

    elif spread_type == "CALL_CREDIT":
        calls = chain[chain.get("side", "") == "call"] if "side" in chain.columns else chain[chain.get("contractSymbol", "").str.contains("CALL", na=False)]
        calls = calls.dropna(subset=["strike"])
        otm_calls = calls[calls["strike"] > current_price]
        if otm_calls.empty:
            return None
        target = current_price * 1.03
        sell_row = otm_calls.iloc[(otm_calls["strike"] - target).abs().argsort()[:1]]
        if sell_row.empty:
            return None
        sell = sell_row.iloc[0]
        buy_candidates = otm_calls[otm_calls["strike"] > sell["strike"]]
        if buy_candidates.empty:
            return None
        desired_buy = sell["strike"] + 5.0
        buy_row = buy_candidates.iloc[(buy_candidates["strike"] - desired_buy).abs().argsort()[:1]]
        if buy_row.empty:
            return None
        buy = buy_row.iloc[0]

        mid_sell = safe_div(sell.get("bid", np.nan) + sell.get("ask", np.nan), 2, default=np.nan)
        mid_buy = safe_div(buy.get("bid", np.nan) + buy.get("ask", np.nan), 2, default=np.nan)
        premium = safe_div(mid_sell - mid_buy, 1, default=np.nan)

        return {
            "type": "Call Credit Spread",
            "expiration": expiration,
            "sell_strike": float(sell["strike"]),
            "buy_strike": float(buy["strike"]),
            "mid_premium": float(premium) if pd.notna(premium) else None,
            "max_profit": round(float(premium * 100) if pd.notna(premium) else 0.0, 2),
            "max_loss": round(float(abs(sell["strike"] - buy["strike"]) * 100 - (premium * 100)) if pd.notna(premium) else 0.0, 2),
            "delta_sell": get_safe_value(sell, "delta"),
            "theta_sell": get_safe_value(sell, "theta"),
            "vega_sell": get_safe_value(sell, "vega"),
        }
    return None

# ---------- Save & report ----------
def save_locally(data: Dict[str, Any], filename: str, local_dir: Path = LOCAL_SAVE_DIR) -> List[Path]:
    """
    Save JSON and Markdown to local_dir and return list of saved Paths.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    json_filename = filename.replace(".md", ".json")
    json_path = local_dir / json_filename
    md_path = local_dir / filename

    saved_files = []
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        saved_files.append(json_path)
        logger.info("Saved JSON: %s", json_path)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(data["markdown_report"])
        saved_files.append(md_path)
        logger.info("Saved Markdown: %s", md_path)
        return saved_files
    except Exception as e:
        logger.exception("Local save failed: %s", e)
        return saved_files

def build_markdown_report(ticker: str, indicators: Dict[str, Any], spreads: List[Dict[str, Any]]) -> str:
    md = f"# AI-Enhanced Credit Spread Analysis: {ticker}\n\n"
    md += f"**Date:** {date.today().isoformat()}\n"
    md += f"**Reference Price:** ${indicators['Current_Price']:.2f}\n\n"
    md += "## Technical Landscape (12 Indicators)\n"
    for k, v in indicators.items():
        if isinstance(v, float):
            if math.isnan(v):
                md += f"- **{k}:** N/A\n"
            else:
                md += f"- **{k}:** {v:.2f}\n"
        else:
            md += f"- **{k}:** {v}\n"
    md += "\n---\n\n"
    if not spreads:
        md += "_No spreads found / generated._\n\n"
    for spread in spreads:
        md += f"### {spread['type']} ({spread['expiration']})\n"
        md += f"**Strategy:** Sell ${spread['sell_strike']} / Buy ${spread['buy_strike']}\n\n"
        md += f"**Est. Credit:** ${spread['max_profit']} | **Max Risk:** ${spread['max_loss']}\n\n"
        md += f"**Analysis:**\n\n{spread.get('rationale', 'No rationale generated.')}\n\n"
        md += "---\n\n"
    return md

# ---------- Main analysis flow ----------
def main(
    ticker_symbol: str = TICKER,
    days_of_history: int = DAYS_OF_HISTORY,
    gcp_bucket: str = GCP_BUCKET,
    mistral_api_key: str = MISTRAL_API_KEY,
    mistral_model: str = MISTRAL_MODEL,
):
    logger.info("Starting analysis for %s", ticker_symbol)

    # Quick Mistral connectivity check
    mistral_ok = quick_mistral_test(mistral_api_key, mistral_model) if mistral_api_key else False
    if not mistral_ok and mistral_api_key:
        logger.warning("Mistral connectivity test returned negative result; AI calls may fail.")

    # Fetch history (limit to days_of_history to be efficient)
    try:
        ticker = yf.Ticker(ticker_symbol)
        # prefer a daily window; use 'period' when available
        df = ticker.history(period=f"{days_of_history}d", auto_adjust=False)
        if df.empty:
            logger.error("No historical data returned for %s", ticker_symbol)
            return
    except Exception as e:
        logger.exception("Failed fetching history: %s", e)
        return

    # Calculate indicators
    try:
        indicators = calculate_technical_indicators(df)
        logger.info("Indicators calculated. Current price: $%.2f", indicators["Current_Price"])
    except Exception as e:
        logger.exception("Indicator calculation failed: %s", e)
        return

    # Get option expirations
    try:
        available_expirations = ticker.options
    except Exception as e:
        logger.exception("Could not fetch option expirations: %s", e)
        available_expirations = []

    if not available_expirations:
        logger.warning("No option expirations returned.")
        # Prepare a report and exit gracefully
        filename = f"{ticker_symbol}_spread_analysis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md"
        md = build_markdown_report(ticker_symbol, indicators, [])
        complete_data = {
            "symbol": ticker_symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "date": date.today().isoformat(),
            "indicators": indicators,
            "spreads": [],
            "markdown_report": md,
        }
        saved = save_locally(complete_data, filename)
        if saved:
            logger.info("Saved minimal report despite no options.")
        return

    # Select expirations near target days
    target_days = [14, 30, 45, 60, 90, 180]
    chosen = []
    for t in target_days:
        target_date = date.today() + timedelta(days=t)
        # available_expirations are strings like '2023-12-15'
        try:
            closest = min(available_expirations, key=lambda x: abs((date.fromisoformat(x) - target_date).days))
            if closest not in chosen:
                chosen.append(closest)
        except Exception:
            continue
    chosen = chosen[:6]
    logger.info("Selected expirations: %s", chosen)

    all_spreads: List[Dict[str, Any]] = []
    api_rate_delay = 1.0  # friendly delay between potentially heavy API calls

    for exp in chosen:
        logger.info("Processing expiration: %s", exp)
        try:
            chain = ticker.option_chain(exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls["side"] = "call"
            puts["side"] = "put"
            full_chain = pd.concat([calls, puts], ignore_index=True, sort=False)

            pcs = select_spread_strikes(full_chain, indicators["Current_Price"], "PUT_CREDIT", exp)
            if pcs:
                # delay to avoid any rate limits
                time.sleep(api_rate_delay)
                pcs["rationale"] = generate_ai_rationale(mistral_api_key, mistral_model, pcs, indicators) if mistral_api_key else "Mistral API not configured."
                all_spreads.append(pcs)

            ccs = select_spread_strikes(full_chain, indicators["Current_Price"], "CALL_CREDIT", exp)
            if ccs:
                time.sleep(api_rate_delay)
                ccs["rationale"] = generate_ai_rationale(mistral_api_key, mistral_model, ccs, indicators) if mistral_api_key else "Mistral API not configured."
                all_spreads.append(ccs)
        except Exception as e:
            logger.exception("Skipping expiration %s due to error: %s", exp, e)
            continue

    # Generate markdown report
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker_symbol}_spread_analysis_{timestamp}.md"
    md_report = build_markdown_report(ticker_symbol, indicators, all_spreads)
    complete_data = {
        "symbol": ticker_symbol,
        "timestamp": timestamp,
        "date": date.today().isoformat(),
        "indicators": indicators,
        "spreads": all_spreads,
        "markdown_report": md_report,
    }

    # Save locally
    saved_files = save_locally(complete_data, filename)
    if not saved_files:
        logger.error("Failed to save any local files. Aborting GCP upload.")
        return

    # Attempt upload with fallback (python client -> gcloud CLI)
    base_prefix = f"options/{ticker_symbol}"
    upload_ok = upload_to_gcp_with_fallback(gcp_bucket, saved_files, base_prefix)
    if upload_ok:
        logger.info("Upload to GCP successful.")
    else:
        logger.error("Upload to GCP failed (both python client & gcloud CLI). Please verify credentials and bucket permissions.")

    logger.info("Analysis complete for %s", ticker_symbol)

# ---------- CLI entrypoint ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Enhanced Options Spread Analysis")
    parser.add_argument("--ticker", type=str, default=TICKER, help="Ticker symbol")
    parser.add_argument("--history-days", type=int, default=DAYS_OF_HISTORY, help="Days of historical data to fetch")
    parser.add_argument("--gcp-bucket", type=str, default=GCP_BUCKET, help="GCP bucket name for uploads")
    parser.add_argument("--mistral-key", type=str, default=os.getenv("MISTRAL_API_KEY", ""), help="Mistral API key (env MISTRAL_API_KEY preferred)")
    parser.add_argument("--mistral-model", type=str, default=MISTRAL_MODEL, help="Mistral model")
    args, unknown = parser.parse_known_args()

    # pass provided args to main
    main(
        ticker_symbol=args.ticker,
        days_of_history=args.history_days,
        gcp_bucket=args.gcp_bucket,
        mistral_api_key=args.mistral_key,
        mistral_model=args.mistral_model,
    )
