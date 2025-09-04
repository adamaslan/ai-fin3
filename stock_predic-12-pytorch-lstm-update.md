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
