
# Parallel Stock Analysis & AI Reporting Workflow

This document outlines a scalable system for processing a large list of stock tickers to fetch data, detect technical signals, and generate AI-driven analysis in parallel.

## 1\. Initialization & Configuration

First, the system is configured with the master list of tickers and the desired data period.

```pseudocode
DEFINE TICKER_LIST = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", ...] // 50+ stocks
DEFINE DATA_PERIOD = "1y"
```

## 2\. Parallel Processing Orchestrator

The main part of the workflow uses a parallel processing pool (like a `ProcessPoolExecutor`) to handle multiple stocks at the same time. Each stock is processed independently on a separate CPU core.

```pseudocode
FUNCTION MainWorkflow:
  // Create a pool of worker processes, one for each CPU core
  CREATE ProcessPool

  // Assign one ticker to each worker process.
  // The 'ProcessTicker' function runs in parallel for each stock.
  results = ProcessPool.map(ProcessTicker, TICKER_LIST)

  // When all parallel jobs are done
  PRINT "Workflow complete."
  SAVE results TO "final_report.json"
END FUNCTION
```

-----

## 3\. Core Logic: `ProcessTicker` (Runs in Parallel)

This is the main function that runs for *each* stock. It follows a four-step pipeline.

```pseudocode
FUNCTION ProcessTicker(ticker):
  PRINT "Starting process for: " + ticker

  // Step 1: Get Data
  dataframe = GetStockData(ticker, DATA_PERIOD)
  IF dataframe IS EMPTY THEN
    RETURN { ticker: ticker, status: "Error", details: "Failed to fetch data." }
  END IF

  // Step 2: Calculate Technical Indicators
  dataframe_with_ta = CalculateIndicators(dataframe)
  IF dataframe_with_ta IS EMPTY THEN
    RETURN { ticker: ticker, status: "Error", details: "Failed to calculate TA." }
  END IF

  // Step 3: Detect Trading Signals
  signals_list = DetectSignals(dataframe_with_ta)

  // Step 4: Generate AI Analysis
  ai_analysis = GetGeminiAnalysis(ticker, signals_list)

  // Step 5: Return the final package of data
  RETURN {
    ticker: ticker,
    status: "Success",
    signals: signals_list,
    analysis: ai_analysis
  }
END FUNCTION
```

-----

## 4\. Sub-Functions

These are the detailed functions called within the `ProcessTicker` pipeline.

### `GetStockData(ticker, period)`

  * Uses the `yfinance` library to fetch the historical OHLCV data for the given `ticker` and `period`.
  * Handles errors if the ticker is invalid or no data is found.
  * Returns a `DataFrame` with the stock's history.

### `CalculateIndicators(dataframe)`

  * Uses the `pandas-ta` library to automatically calculate and append a wide set of indicators to the `DataFrame`.
  * Returns the modified `DataFrame` with new columns (e.g., `SMA_20`, `RSI_14`, `MACD_12_26_9`, `BBU_20_2.0`, etc.).

### `DetectSignals(dataframe_with_ta)`

This is the core custom logic engine where all 200+ alerts are defined. It checks the *most recent* data row against the signal criteria.

```pseudocode
FUNCTION DetectSignals(dataframe_with_ta):
  // Get the last two rows for crossover/momentum checks
  latest_row = GET last_row FROM dataframe_with_ta
  previous_row = GET second_to_last_row FROM dataframe_with_ta

  CREATE empty signals_found_list

  // --- Example Signal Logic ---

  // Signal: RSI Oversold
  IF latest_row.RSI_14 < 30 THEN
    ADD "RSI Oversold" TO signals_found_list
  END IF

  // Signal: RSI Overbought
  IF latest_row.RSI_14 > 70 THEN
    ADD "RSI Overbought" TO signals_found_list
  END IF

  // Signal: Golden Cross (SMA 20/50)
  IF latest_row.SMA_20 > latest_row.SMA_50 AND previous_row.SMA_20 <= previous_row.SMA_50 THEN
    ADD "Golden Cross (SMA 20/50)" TO signals_found_list
  END IF

  // Signal: Price above Upper Bollinger Band
  IF latest_row.Close > latest_row.BBU_20_2.0 THEN
    ADD "Price Above Upper Bollinger Band" TO signals_found_list
  END IF

  // ... (Add 200+ more signal checks here) ...

  IF signals_found_list IS EMPTY THEN
    ADD "No significant signals detected." TO signals_found_list
  END IF

  RETURN signals_found_list
END FUNCTION
```

### `GetGeminiAnalysis(ticker, signals_list)`

This function uses generative AI to create a human-readable summary of the findings.

```pseudocode
FUNCTION GetGeminiAnalysis(ticker, signals_list):
  IF signals_list CONTAINS "No significant signals detected." THEN
    RETURN "No analysis required."
  END IF

  // Build the prompt for the AI
  prompt = "Act as a concise financial analyst.
            Stock: " + ticker + "
            Detected Signals: " + join(signals_list, ", ") + "
            Provide a 2-3 sentence analysis of what these signals mean when combined."

  // Call the Google Gemini API
  CONNECT to Google AI with API_KEY
  response = CALL GeminiModel.generate_content(prompt)

  RETURN response.text
END FUNCTION
```