/# 🧩 20-Point Pipeline for Stock Prediction with Sentiment Integration

This pipeline extends the `GenericStockTargetPredictor` to incorporate **SPY sentiment features** (average sentiment + volatility).  
It shows how the existing PyTorch + LSTM code changes step by step.

---

## 📂 Data Acquisition
1. **Load API Keys & Environment Variables** – Secure Alpha Vantage + sentiment provider keys via `.env`.  
2. **Fetch Historical Stock Data** – Use Alpha Vantage daily time series for the target stock (e.g., AAPL).  
3. **Fetch SPY Sentiment Data** – Pull news sentiment scores and daily volatility for SPY from a sentiment API.  
4. **Align Stock & Sentiment Timelines** – Merge stock OHLCV with daily SPY sentiment, forward-filling missing dates.  

---

## 📊 Feature Engineering
5. **Technical Indicators** – Compute returns, rolling volatility, RSI, SMA ratio, volume ratio.  
6. **Sentiment Indicators** – Add `sentiment_avg` and `sentiment_volatility` as new columns.  
7. **Feature Scaling** – Apply `StandardScaler` across both technical and sentiment features to balance magnitudes.  
8. **Feature Selection** – Expand feature list from 5 → 7:  
   - Returns  
   - Volatility  
   - RSI  
   - SMA Ratio  
   - Volume Ratio  
   - Sentiment Avg  
   - Sentiment Volatility  

---

## 🧮 Data Preparation
9. **Sequence Creation** – Build sliding windows (`lookback_days`) including sentiment features.  
10. **Target Labeling** – Classify each sequence for upward/downward moves based on target % thresholds.  
11. **Dataset Wrapping** – Convert to `StockDataset` with sequences (X) and targets (y).  
12. **Batching** – Use PyTorch `DataLoader` to shuffle and batch training data.  

---

## 🧠 Model Architecture
13. **LSTM Input Layer** – Update `input_features=7` to match the expanded feature set.  
14. **Hidden State Processing** – LSTM captures temporal patterns across technical + sentiment interactions.  
15. **Dropout Regularization** – Apply dropout in both LSTM and classifier layers to avoid overfitting on sentiment noise.  
16. **Classifier Head** – Map LSTM outputs through dense layers → sigmoid probability output.  

---

## ⚙️ Training & Optimization
17. **Loss Function** – Use `nn.BCELoss` to optimize probability predictions for binary targets.  
18. **Optimizer** – Adam optimizer with learning rate tuning (0.001 default).  
19. **Training Loop** – Forward pass, loss computation, backpropagation, and weight updates per epoch.  
20. **Validation & Prediction** – Evaluate latest sequence with sentiment features → produce probability estimates for ±1% to ±10% targets.  

---

## 🚀 Outputs
- Probabilities of reaching each upward/downward target.  
- Visualization of probabilities vs. targets with sentiment-aware adjustments.  
- Highlighted best upward and downward scenarios.  


# 🔗 12 Ways Sentiment Integration Affects PyTorch + LSTM

## 🔄 Data Handling & Features
1. **Feature Dimensionality Increase** – Input feature size goes from 5 → 7, so the `LSTM(input_size, hidden_size)` in PyTorch automatically handles the extra sentiment features.  
2. **Normalization with StandardScaler** – Sentiment features (avg + volatility) are scaled alongside price-based features, preventing one feature (e.g., raw sentiment score) from dominating training.  
3. **Time-Series Alignment** – The constant sentiment values (applied across rows) become part of each sequence window, feeding into the recurrent dynamics of the LSTM.  
4. **Batch Processing in DataLoader** – Sentiment-enhanced sequences are wrapped in `StockDataset` and loaded into batches just like price features, ensuring training speed stays consistent.  

## 🧠 LSTM Dynamics
5. **Hidden State Encoding** – The LSTM hidden states capture interactions between technical indicators (RSI, volatility) and sentiment features, learning relationships like “high SPY sentiment + low AAPL RSI = bullish setup.”  
6. **Temporal Smoothing** – Since news sentiment is less noisy than price ticks, the LSTM integrates it over time, helping reduce prediction volatility.  
7. **Dropout Regularization** – Existing LSTM dropout layers prevent overfitting when sentiment data is sparse or noisy, keeping generalization intact.  
8. **Gradient Flow** – Backpropagation updates LSTM weights to adjust how much sentiment signals influence predictions compared to technical signals.  

## 📊 Classification Layer Effects
9. **Enhanced Linear Transformation** – The `nn.Linear(hidden_size → 16)` layer after the LSTM maps richer representations that now include sentiment-aware states.  
10. **Nonlinear Interactions** – The ReLU activation combines price-driven and sentiment-driven patterns nonlinearly, allowing the model to detect asymmetric effects (e.g., bullish sentiment may impact upward targets more than downward).  

## ⚙️ Training & Predictions
11. **Loss Function Sensitivity** – With `nn.BCELoss`, the model is penalized not just for misclassifying price patterns but also for ignoring sentiment-driven shifts, reinforcing sentiment’s role.  
12. **Probability Calibration** – The sigmoid output layer now encodes a blend of market technicals + SPY sentiment, giving more realistic probabilities of AAPL moving ±X%.  

# 🔗 12 Ways Sentiment Integration Affects PyTorch + LSTM

## 🔄 Data Handling & Features
1. **Feature Dimensionality Increase** – Input feature size goes from 5 → 7, so the `LSTM(input_size, hidden_size)` in PyTorch automatically handles the extra sentiment features.  
2. **Normalization with StandardScaler** – Sentiment features (avg + volatility) are scaled alongside price-based features, preventing one feature (e.g., raw sentiment score) from dominating training.  
3. **Time-Series Alignment** – The constant sentiment values (applied across rows) become part of each sequence window, feeding into the recurrent dynamics of the LSTM.  
4. **Batch Processing in DataLoader** – Sentiment-enhanced sequences are wrapped in `StockDataset` and loaded into batches just like price features, ensuring training speed stays consistent.  

## 🧠 LSTM Dynamics
5. **Hidden State Encoding** – The LSTM hidden states capture interactions between technical indicators (RSI, volatility) and sentiment features, learning relationships like “high SPY sentiment + low AAPL RSI = bullish setup.”  
6. **Temporal Smoothing** – Since news sentiment is less noisy than price ticks, the LSTM integrates it over time, helping reduce prediction volatility.  
7. **Dropout Regularization** – Existing LSTM dropout layers prevent overfitting when sentiment data is sparse or noisy, keeping generalization intact.  
8. **Gradient Flow** – Backpropagation updates LSTM weights to adjust how much sentiment signals influence predictions compared to technical signals.  

## 📊 Classification Layer Effects
9. **Enhanced Linear Transformation** – The `nn.Linear(hidden_size → 16)` layer after the LSTM maps richer representations that now include sentiment-aware states.  
10. **Nonlinear Interactions** – The ReLU activation combines price-driven and sentiment-driven patterns nonlinearly, allowing the model to detect asymmetric effects (e.g., bullish sentiment may impact upward targets more than downward).  

## ⚙️ Training & Predictions
11. **Loss Function Sensitivity** – With `nn.BCELoss`, the model is penalized not just for misclassifying price patterns but also for ignoring sentiment-driven shifts, reinforcing sentiment’s role.  
12. **Probability Calibration** – The sigmoid output layer now encodes a blend of market technicals + SPY sentiment, giving more realistic probabilities of AAPL moving ±X%.  
