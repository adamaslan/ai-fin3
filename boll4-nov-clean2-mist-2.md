# to do
## limit tokens created to shorten responses

# 50-Function Summary of the AI Technical Analysis Script

## 1. **Environment & Configuration**
1. Loads environment variables from `.env` file
2. Configures Mistral API with model selection
3. Validates API key availability
4. Sets ticker symbol and history duration

## 2. **Data Handling & Utilities**
5. Custom JSON encoder for NumPy data types
6. Safe value extraction from pandas rows
7. Date manipulation for expiration targeting
8. File path management for outputs

## 3. **API Communication**
9. Performs API connectivity test
10. Implements retry logic with exponential backoff
11. Handles rate limiting (429 errors)
12. Processes API response validation
13. Manages API call delays between requests

## 4. **Data Acquisition**
14. Fetches historical price data via Yahoo Finance
15. Downloads options chain data for multiple expirations
16. Validates data availability and completeness

## 5. **Technical Indicators (12 Calculated)**
17. 50-day Simple Moving Average (SMA)
18. 200-day Simple Moving Average
19. Relative Strength Index (RSI 14)
20. MACD with signal line
21. Bollinger Bands with %B indicator
22. Historical Volatility (30-day)
23. Average True Range (ATR 14)
24. 50-day Average Volume
25. 10-day Rate of Change (ROC)
26. Stochastic Oscillator %K
27. Money Flow Index (MFI)
28. Accumulation/Distribution Line

## 6. **Options Analysis**
29. Filters option chains by strike proximity
30. Calculates mid-point prices from bid/ask
31. Determines credit spread premiums
32. Computes maximum profit/loss for spreads
33. Selects appropriate strike distances (typically $5 apart)
34. Handles both put and call credit spreads

## 7. **AI Integration**
35. Constructs detailed prompt with technical context
36. Formats trade structure for AI analysis
37. Generates sophisticated market analysis using Mistral
38. Includes risk disclosure in all AI outputs
39. Manages API token usage and parameters

## 8. **Report Generation**
40. Creates timestamp-based unique filenames
41. Generates comprehensive markdown reports
42. Saves raw analysis data as JSON
43. Creates Next.js compatible structured JSON
44. Organizes output into separate directories
45. Formats technical indicators for readability
46. Structures trade details with clear labeling

## 9. **Error Handling & Validation**
47. Validates data availability at each step
48. Handles missing Greek data gracefully
49. Manages empty option chain scenarios
50. Provides informative console feedback throughout execution

## **Key Features Summary:**
- **Multi-timeframe analysis**: 6 expiration dates (14-180 days)
- **Dual strategy generation**: Both put and call credit spreads
- **AI-enhanced rationale**: Context-aware trade analysis
- **Multiple output formats**: Markdown, raw JSON, and Next.js JSON
- **Comprehensive technical analysis**: 12 indicators for market context
- **Professional risk metrics**: Calculates max profit/loss, Greeks when available
- **Rate limit protection**: Exponential backoff for API calls
- **Production-ready structure**: Organized output and error handling