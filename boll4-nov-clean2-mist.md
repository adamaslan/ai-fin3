# Code Explanation: AI-Enhanced Options Analysis Tool

## Overview
This Python script performs technical analysis on a stock and generates hypothetical options credit spread strategies with AI-powered rationales using the Mistral API.

## Function-by-Function Breakdown

### 1. **Configuration & Initialization**
```python
# Global constants and API setup
TICKER = "MP"  # Target stock symbol
DAYS_OF_HISTORY = 90  # Historical data window
MISTRAL_MODEL = "mistral-small-latest"  # AI model to use
```

### 2. **`quick_api_test()`**
**Purpose**: Validates Mistral API connectivity before proceeding with analysis.

**Key Features**:
- Tests API connection with a minimal prompt
- Uses 15 API parameters for completeness (stream, temperature, max_tokens, etc.)
- Sets `API_CONFIGURED = False` if test fails to prevent subsequent AI calls
- Provides clear success/error messages

### 3. **`calculate_technical_indicators(df)`**
**Purpose**: Computes 12 technical indicators from historical price data.

**Calculated Indicators**:
1. **SMA_50/SMA_200**: Simple Moving Averages (50/200-day)
2. **RSI**: Relative Strength Index (14-day)
3. **MACD**: Moving Average Convergence Divergence with Signal Line
4. **Bollinger Bands**: Upper/Lower bands and %B indicator
5. **Historical Volatility**: 30-day annualized volatility
6. **ATR**: Average True Range (14-day)
7. **Average Volume**: 50-day moving average of volume
8. **ROC**: Rate of Change (10-day percentage)
9. **Stochastic Oscillator**: %K value (14-day)
10. **MFI**: Money Flow Index (14-day)
11. **A/D Line**: Accumulation/Distribution Line

**Returns**: Dictionary with latest values of all indicators

### 4. **`get_safe_value(row, col_name)`**
**Purpose**: Safe accessor for DataFrame columns that might not exist.

**Why needed**: Option chain data from yFinance may have missing Greek columns (delta, theta, vega). This function prevents KeyError exceptions.

### 5. **`select_spread_strikes(chain, current_price, spread_type, expiration)`**
**Purpose**: Selects strike prices for credit spread strategies.

**Spread Types**:
- **Put Credit Spread** (Bullish/Neutral): Sell OTM put, buy further OTM put
- **Call Credit Spread** (Bearish/Neutral): Sell OTM call, buy further OTM call

**Selection Logic**:
1. Filters options chain by type (put/call)
2. Selects sell strike ~2-5% out-of-the-money from current price
3. Selects buy strike $5 away from sell strike (or closest available)
4. Calculates mid-price premium using bid/ask averages
5. Computes max profit/loss based on spread width and premium

**Returns**: Dictionary with trade structure details including Greeks (if available)

### 6. **`generate_ai_rationale(spread_data, indicators)`**
**Purpose**: Generates AI analysis of spread strategy using technical indicators.

**Key Features**:
- **Exponential Backoff**: Implements retry logic with 2ⁿ delay for rate limits
- **15 API Parameters**: Uses comprehensive parameter set for optimal output:
  - `temperature=0.5`: Balanced creativity/determinism
  - `max_tokens=800`: Adequate for detailed analysis
  - `frequency_penalty=0.1`: Reduces word repetition
  - `presence_penalty=0.1`: Encourages topic variety
  - `random_seed=42`: Ensures reproducibility
- **Context-Rich Prompt**: Provides all technical indicators and trade details
- **Safety Measures**: Includes disclaimer about hypothetical nature

**Prompt Structure**:
1. Market context with 12 technical indicators
2. Trade structure details
3. Instructions for connecting indicators to trade thesis

### 7. **`main_analysis()`**
**Purpose**: Main orchestration function that ties everything together.

**Workflow**:
1. **API Test**: Calls `quick_api_test()` to verify connectivity
2. **Data Fetch**: Retrieves historical price data using yFinance
3. **Indicator Calculation**: Computes technical indicators
4. **Expiration Selection**: Chooses 6 option expirations (14-180 days out)
5. **Spread Analysis**: For each expiration:
   - Gets options chain
   - Generates both put and call credit spreads
   - Requests AI rationale with 3-second delay between API calls
6. **Report Generation**: Creates markdown report with:
   - Technical indicator summary
   - Each spread strategy with AI rationale
   - Risk/reward metrics

**Output**: Saves to `spreads-yo/{TICKER}_spread_analysis_{timestamp}.md`

## Key Design Patterns

### 1. **Graceful Degradation**
- API failures don't crash the program
- Missing Greek columns are handled gracefully
- Empty data returns informative messages

### 2. **Rate Limiting Protection**
- 3-second delay between AI calls
- Exponential backoff for rate limit errors (2, 4, 8 seconds)

### 3. **Comprehensive Error Handling**
- Try/except blocks around API calls
- Validation for empty data structures
- Clear user feedback on failures

### 4. **Modular Design**
Each function has a single responsibility:
- Data calculation (`calculate_technical_indicators`)
- Strategy selection (`select_spread_strikes`)
- AI interaction (`generate_ai_rationale`)
- Orchestration (`main_analysis`)

## File Structure Created
```
spreads-yo/
└── {TICKER}_spread_analysis_{timestamp}.md
```

## Dependencies
- **yfinance**: Stock and options data
- **pandas/numpy**: Data manipulation
- **mistralai**: Official Mistral SDK
- **python-dotenv**: Environment variable management

## Use Cases
1. **Options Traders**: Generate strategy ideas with AI analysis
2. **Technical Analysts**: Comprehensive indicator calculation
3. **Educational Tool**: Learn options strategies with AI explanations
4. **Research**: Backtest hypothetical strategies

## Limitations
- **Hypothetical Only**: Uses mid-prices, not executable orders
- **Data Delays**: yFinance data may be delayed
- **AI Hallucinations**: AI rationales may contain inaccuracies
- **Rate Limited**: API calls are throttled to prevent exceeding limits

## Security Notes
- API keys loaded from environment variables (`.env` file)
- No hardcoded credentials
- Safe prompt disabled for financial analysis context