# 📊 Enhancements: Integrating Market News & Sentiment into Stock Predictor

This document outlines **42 structured changes** to extend the
`GenericStockTargetPredictor` with **Alpha Vantage News & Sentiment
data** (e.g., SPY sentiment) to improve predictions of AAPL's
probability of price movement.

------------------------------------------------------------------------

## 🔑 API Integration

1.  Use Alpha Vantage `NEWS_SENTIMENT` function.\
2.  Add a new method `get_news_sentiment()` in the predictor class.\
3.  Support `tickers` parameter (default `"SPY"`).\
4.  Use `limit` parameter (default `50`) to fetch enough articles.\
5.  Parse the `"feed"` array from the API response.\
6.  Extract `"ticker_sentiment_score"` for the target ticker.\
7.  Handle cases when `"feed"` is missing (raise exception).\
8.  Handle empty sentiment results (default to `0.0`).\
9.  Return both **average sentiment** and **sentiment volatility**.\
10. Log results with ✓ confirmation (for debugging).

------------------------------------------------------------------------

## 📈 Feature Engineering

11. Modify `calculate_features()` to call `get_news_sentiment()`.\
12. Add a new feature column: `news_sentiment` (avg SPY sentiment).\
13. Add another feature: `news_sent_vol` (SPY sentiment volatility).\
14. Keep sentiment values constant across rows (applies to latest
    news).\
15. Fill NaN values with `ffill` and `0`.\
16. Update `feature_cols` list in `create_training_data()`.\
17. Ensure sentiment integrates smoothly with price features.\
18. Retain backward compatibility if sentiment fetch fails.\
19. Allow future expansion to multiple tickers.\
20. Allow topic filters (e.g., `"economy_macro"`) in future.

------------------------------------------------------------------------

## 🧠 Model Training

21. Increase input feature size from 5 → 7.\
22. Verify `StockPredictor` LSTM handles new features automatically.\
23. Keep `hidden_size=32` but allow scaling if needed.\
24. Ensure `StandardScaler` normalizes sentiment features.\
25. Check for feature dominance (sentiment ranges vs. price features).\
26. Run additional training epochs if sentiment is noisy.\
27. Save sentiment-enhanced models separately if needed.\
28. Add option to disable sentiment features (debug mode).\
29. Test both SPY-only and mixed sentiment signals.\
30. Evaluate training stability with added features.

------------------------------------------------------------------------

## 📊 Prediction Results

31. Include sentiment-based features in probability predictions.\
32. Re-train for all upward/downward target ranges.\
33. Store predictions in the same results dictionary.\
34. Historical success metric remains unchanged (price-only).\
35. Compare model probability shifts with and without sentiment.\
36. Highlight sentiment impact in log messages.\
37. Extend `display_results()` with optional sentiment summary.\
38. Add "market risk index" (probability of ±5% moves).\
39. Highlight correlation between SPY sentiment and AAPL probabilities.\
40. Use sentiment to interpret **best upward** and **best downward**
    targets.\
41. In `create_visualization()`, allow overlay of sentiment level.\
42. Save enhanced plots as
    `{symbol}_sentiment_target_probabilities.png`.

------------------------------------------------------------------------

✅ With these 42 changes, the predictor evolves from a **purely
technical model** into a **hybrid technical + sentiment-aware
predictor**, capturing both **price action** and **macro news-driven
market risk**.
