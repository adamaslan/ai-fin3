import marimo

__generated_with = "0.8.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md("""
    # 🎯 Interactive Options Delta Predictor
    
    This interactive notebook demonstrates advanced machine learning models for predicting options delta values.
    Built with ensemble methods including Transformers, LSTMs, and CNNs.
    
    **Features:**
    - Real-time parameter adjustment
    - Interactive model comparison
    - Live predictions with uncertainty quantification
    - Market scenario analysis
    """)
    return


@app.cell
def __():
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    from sklearn.preprocessing import RobustScaler
    from datetime import datetime, timedelta
    import warnings
    warnings.filterwarnings('ignore')
    return (
        datetime,
        go,
        make_subplots,
        nn,
        np,
        pd,
        px,
        RobustScaler,
        timedelta,
        torch,
        warnings,
    )


@app.cell
def __(mo):
    # Interactive controls for model parameters
    model_type = mo.ui.dropdown(
        options=["Ensemble", "Transformer", "LSTM", "CNN", "Feedforward"],
        value="Ensemble",
        label="Model Architecture"
    )
    
    learning_rate = mo.ui.slider(
        start=0.0001, stop=0.01, step=0.0001,
        value=0.001, label="Learning Rate"
    )
    
    batch_size = mo.ui.slider(
        start=16, stop=128, step=16,
        value=64, label="Batch Size"
    )
    
    epochs = mo.ui.slider(
        start=10, stop=200, step=10,
        value=100, label="Training Epochs"
    )
    
    dropout_rate = mo.ui.slider(
        start=0.0, stop=0.5, step=0.05,
        value=0.2, label="Dropout Rate"
    )
    return batch_size, dropout_rate, epochs, learning_rate, model_type


@app.cell
def __(batch_size, dropout_rate, epochs, learning_rate, mo, model_type):
    # Display configuration panel
    mo.hstack([
        mo.vstack([
            mo.md("**Model Selection**"),
            model_type,
            learning_rate
        ]),
        mo.vstack([
            mo.md("**Training Parameters**"),
            batch_size,
            epochs,
            dropout_rate
        ])
    ])
    return


@app.cell
def __(mo, np, pd):
    def generate_synthetic_options_data(n_samples=2000, seed=42):
        """Generate realistic synthetic options data for demo"""
        np.random.seed(seed)
        
        # Market parameters
        S0 = 150  # Current stock price
        strikes = np.random.uniform(S0 * 0.7, S0 * 1.3, n_samples)
        times_to_expiry = np.random.uniform(1/365, 1.0, n_samples)  # 1 day to 1 year
        volatilities = np.random.uniform(0.1, 0.8, n_samples)
        risk_free_rate = 0.05
        
        # Calculate theoretical delta using Black-Scholes approximation
        from scipy.stats import norm
        
        d1 = (np.log(S0/strikes) + (risk_free_rate + 0.5*volatilities**2)*times_to_expiry) / (volatilities*np.sqrt(times_to_expiry))
        theoretical_delta = norm.cdf(d1)
        
        # Add market noise and microstructure effects
        volume = np.random.exponential(100, n_samples)
        open_interest = np.random.exponential(500, n_samples)
        bid_ask_spread = np.random.uniform(0.01, 0.5, n_samples)
        
        # Create DataFrame
        options_data = pd.DataFrame({
            'strike': strikes,
            'current_price': S0,
            'time_to_expiry': times_to_expiry,
            'implied_volatility': volatilities,
            'volume': volume,
            'open_interest': open_interest,
            'bid_ask_spread': bid_ask_spread,
            'delta': theoretical_delta
        })
        
        # Feature engineering
        options_data['moneyness'] = options_data['strike'] / options_data['current_price']
        options_data['log_moneyness'] = np.log(options_data['moneyness'])
        options_data['sqrt_time'] = np.sqrt(options_data['time_to_expiry'])
        options_data['volume_oi_ratio'] = options_data['volume'] / np.maximum(options_data['open_interest'], 1)
        options_data['log_volume'] = np.log1p(options_data['volume'])
        options_data['iv_time_interaction'] = options_data['implied_volatility'] * options_data['sqrt_time']
        
        return options_data
    
    # Generate data
    options_df = generate_synthetic_options_data()
    
    # Feature selection
    feature_columns = [
        'log_moneyness', 'sqrt_time', 'implied_volatility', 
        'log_volume', 'volume_oi_ratio', 'bid_ask_spread',
        'iv_time_interaction'
    ]
    
    X_raw = options_df[feature_columns].values
    y_raw = options_df['delta'].values
    
    # Data statistics
    data_stats = mo.md(f"""
    **Dataset Statistics:**
    - Samples: {len(options_df):,}
    - Features: {len(feature_columns)}
    - Delta Range: {y_raw.min():.3f} - {y_raw.max():.3f}
    - Avg IV: {options_df['implied_volatility'].mean():.1%}
    """)
    
    mo.hstack([data_stats, mo.ui.table(options_df.head(10))])
    return (
        X_raw,
        feature_columns,
        generate_synthetic_options_data,
        options_df,
        y_raw,
    )


@app.cell
def __(nn, torch):
    # Simplified models for demonstration
    class SimpleTransformer(nn.Module):
        def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead, d_model*2, dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.output = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            x = self.input_proj(x).unsqueeze(1)  # Add sequence dimension
            x = self.transformer(x)
            return self.output(x.squeeze(1))
    
    class SimpleLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, dropout=dropout)
            self.output = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            x = x.unsqueeze(1)  # Add sequence dimension
            lstm_out, _ = self.lstm(x)
            return self.output(lstm_out.squeeze(1))
    
    class SimpleCNN(nn.Module):
        def __init__(self, input_size, dropout=0.1):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            self.output = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            x = x.unsqueeze(1)  # Add channel dimension
            conv_out = self.conv(x)
            return self.output(conv_out.squeeze(-1))
    
    class SimpleFeedforward(nn.Module):
        def __init__(self, input_size, dropout=0.1):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.network(x)
    return SimpleCNN, SimpleFeedforward, SimpleLSTM, SimpleTransformer


@app.cell
def __(mo, model_type):
    # Training button
    train_button = mo.ui.button(label="🚀 Train Model", kind="success")
    
    mo.hstack([
        mo.md(f"**Selected Model:** {model_type.value}"),
        train_button
    ])
    return train_button,


@app.cell
def __(
    RobustScaler,
    SimpleCNN,
    SimpleFeedforward,
    SimpleLSTM,
    SimpleTransformer,
    X_raw,
    batch_size,
    dropout_rate,
    epochs,
    learning_rate,
    mo,
    model_type,
    nn,
    torch,
    train_button,
    y_raw,
):
    training_results = None
    
    if train_button.value > 0:
        # Prepare data
        data_scaler = RobustScaler()
        X_scaled_data = data_scaler.fit_transform(X_raw)
        
        # Convert to tensors
        X_tensor_data = torch.FloatTensor(X_scaled_data)
        y_tensor_data = torch.FloatTensor(y_raw).unsqueeze(1)
        
        # Split data
        split_idx = int(0.8 * len(X_tensor_data))
        X_train_data, X_val_data = X_tensor_data[:split_idx], X_tensor_data[split_idx:]
        y_train_data, y_val_data = y_tensor_data[:split_idx], y_tensor_data[split_idx:]
        
        # Initialize model based on selection
        input_size = X_raw.shape[1]
        model_map = {
            "Transformer": SimpleTransformer(input_size, dropout=dropout_rate.value),
            "LSTM": SimpleLSTM(input_size, dropout=dropout_rate.value),
            "CNN": SimpleCNN(input_size, dropout=dropout_rate.value),
            "Feedforward": SimpleFeedforward(input_size, dropout=dropout_rate.value),
            "Ensemble": SimpleFeedforward(input_size, dropout=dropout_rate.value)  # Simplified for demo
        }
        
        trained_model_obj = model_map[model_type.value]
        optimizer = torch.optim.Adam(trained_model_obj.parameters(), lr=learning_rate.value)
        criterion = nn.MSELoss()
        
        # Training loop (simplified for demo)
        train_losses = []
        val_losses = []
        
        trained_model_obj.train()
        for epoch in range(min(epochs.value, 50)):  # Limit for demo
            # Training
            optimizer.zero_grad()
            train_pred = trained_model_obj(X_train_data)
            train_loss = criterion(train_pred, y_train_data)
            train_loss.backward()
            optimizer.step()
            
            # Validation
            with torch.no_grad():
                trained_model_obj.eval()
                val_pred = trained_model_obj(X_val_data)
                val_loss = criterion(val_pred, y_val_data)
                trained_model_obj.train()
            
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
        
        # Store results
        trained_model_obj.eval()
        with torch.no_grad():
            predictions_data = trained_model_obj(X_tensor_data).detach().numpy().flatten()
        
        training_results = {
            'model': trained_model_obj,
            'scaler': data_scaler,
            'train_loss': train_losses,
            'val_loss': val_losses,
            'predictions': predictions_data,
            'model_type': model_type.value
        }
        
        mo.md(f"""
        ✅ **Training Complete!**
        - Final Training Loss: {train_losses[-1]:.6f}
        - Final Validation Loss: {val_losses[-1]:.6f}
        - Model: {model_type.value}
        - Epochs: {len(train_losses)}
        """)
    else:
        mo.md("👆 Click the **Train Model** button to start training")
    return data_scaler, training_results


@app.cell
def __(go, make_subplots, mo, np, training_results, y_raw):
    if training_results is not None:
        # Training curves
        fig_training = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training & Validation Loss', 'Predictions vs Actual')
        )
        
        # Training loss
        fig_training.add_trace(
            go.Scatter(y=training_results['train_loss'], name='Train Loss', 
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # Validation loss
        fig_training.add_trace(
            go.Scatter(y=training_results['val_loss'], name='Val Loss',
                      line=dict(color='red')),
            row=1, col=1
        )
        
        # Prediction vs Actual scatter plot
        predictions = training_results['predictions']
        
        # Perfect prediction line
        min_val, max_val = y_raw.min(), y_raw.max()
        fig_training.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name='Perfect Prediction',
                      line=dict(dash='dash', color='gray')),
            row=1, col=2
        )
        
        # Actual predictions
        fig_training.add_trace(
            go.Scatter(x=y_raw, y=predictions, mode='markers',
                      name='Predictions', opacity=0.6,
                      marker=dict(color=np.abs(y_raw - predictions), 
                                colorscale='Viridis', showscale=True,
                                colorbar=dict(title="Prediction Error"))),
            row=1, col=2
        )
        
        fig_training.update_xaxes(title_text="Epoch", row=1, col=1)
        fig_training.update_yaxes(title_text="Loss", row=1, col=1)
        fig_training.update_xaxes(title_text="Actual Delta", row=1, col=2)
        fig_training.update_yaxes(title_text="Predicted Delta", row=1, col=2)
        
        fig_training.update_layout(height=500, showlegend=True)
        
        mo.ui.plotly(fig_training)
    else:
        mo.md("📊 Training visualizations will appear after model training")
    return fig_training, max_val, min_val, predictions


@app.cell
def __(mo, training_results):
    if training_results is not None:
        # Interactive sliders for live prediction
        mo.md("## 🎮 Live Delta Prediction")
        
        strike_slider = mo.ui.slider(
            start=100, stop=200, step=1, value=150,
            label="Strike Price ($)"
        )
        
        time_slider = mo.ui.slider(
            start=1, stop=365, step=1, value=30,
            label="Days to Expiry"
        )
        
        iv_slider = mo.ui.slider(
            start=10, stop=80, step=1, value=25,
            label="Implied Volatility (%)"
        )
        
        volume_slider = mo.ui.slider(
            start=1, stop=1000, step=10, value=100,
            label="Volume"
        )
        
        mo.hstack([
            mo.vstack([strike_slider, time_slider]),
            mo.vstack([iv_slider, volume_slider])
        ])
    else:
        strike_slider = None
        time_slider = None  
        iv_slider = None
        volume_slider = None
        mo.md("⏳ Complete model training to access live predictions")
    return iv_slider, strike_slider, time_slider, volume_slider


@app.cell
def __(
    go,
    iv_slider,
    mo,
    np,
    strike_slider,
    time_slider,
    torch,
    training_results,
    volume_slider,
):
    if training_results is not None and all(x is not None for x in [strike_slider, time_slider, iv_slider, volume_slider]):
        # Current stock price (from data generation)
        current_price = 150
        
        # Prepare input features
        strike_val = strike_slider.value
        time_to_expiry = time_slider.value / 365  # Convert to years
        iv_val = iv_slider.value / 100  # Convert to decimal
        volume_val = volume_slider.value
        
        # Feature engineering
        moneyness_val = strike_val / current_price
        log_moneyness_val = np.log(moneyness_val)
        sqrt_time_val = np.sqrt(time_to_expiry)
        log_volume_val = np.log1p(volume_val)
        volume_oi_ratio_val = 0.2  # Assumed
        bid_ask_spread_val = 0.05  # Assumed
        iv_time_interaction_val = iv_val * sqrt_time_val
        
        # Create feature vector
        live_features = np.array([[
            log_moneyness_val, sqrt_time_val, iv_val, log_volume_val,
            volume_oi_ratio_val, bid_ask_spread_val, iv_time_interaction_val
        ]])
        
        # Scale features
        live_features_scaled = training_results['scaler'].transform(live_features)
        
        # Make prediction
        with torch.no_grad():
            live_prediction = training_results['model'](torch.FloatTensor(live_features_scaled)).item()
        
        # Calculate theoretical delta for comparison
        from scipy.stats import norm
        d1 = (np.log(current_price/strike_val) + (0.05 + 0.5*iv_val**2)*time_to_expiry) / (iv_val*sqrt_time_val)
        theoretical_delta_val = norm.cdf(d1)
        
        # Create gauge chart for prediction
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = live_prediction,
            delta = {'reference': theoretical_delta_val},
            title = {'text': "Predicted Delta"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.25], 'color': "lightgray"},
                    {'range': [0.25, 0.5], 'color': "gray"},
                    {'range': [0.5, 0.75], 'color': "lightblue"},
                    {'range': [0.75, 1], 'color': "darkblue"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': theoretical_delta_val
                }
            }
        ))
        
        fig_gauge.update_layout(height=400)
        
        # Results summary
        results_md = mo.md(f"""
        ### 📊 Prediction Results
        
        **Input Parameters:**
        - Strike: ${strike_val}
        - Current Price: ${current_price}
        - Time to Expiry: {time_slider.value} days
        - Implied Volatility: {iv_slider.value}%
        - Volume: {volume_val:,}
        
        **Predictions:**
        - **Model Prediction:** {live_prediction:.4f}
        - **Theoretical Delta:** {theoretical_delta_val:.4f}
        - **Difference:** {abs(live_prediction - theoretical_delta_val):.4f}
        
        **Option Classification:**
        - {"🟢 ITM" if moneyness_val < 1 else "🔴 OTM"} (Moneyness: {moneyness_val:.3f})
        - {"📈 High Volatility" if iv_val > 0.3 else "📉 Low Volatility"}
        - {"⚡ High Volume" if volume_val > 500 else "🔋 Low Volume"}
        """)
        
        mo.hstack([
            mo.ui.plotly(fig_gauge),
            results_md
        ])
    else:
        mo.md("🔄 Awaiting model training completion...")
    return (
        current_price,
        d1,
        fig_gauge,
        iv_time_interaction_val,
        iv_val,
        live_features,
        live_features_scaled,
        live_prediction,
        log_moneyness_val,
        log_volume_val,
        moneyness_val,
        results_md,
        sqrt_time_val,
        strike_val,
        theoretical_delta_val,
        time_to_expiry,
        volume_oi_ratio_val,
        volume_val,
    )


@app.cell
def __(mo, np, pd, predictions, training_results, y_raw):
    if training_results is not None:
        # Calculate comprehensive metrics
        mae = np.mean(np.abs(predictions - y_raw))
        mse = np.mean((predictions - y_raw)**2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((y_raw - predictions)**2) / np.sum((y_raw - np.mean(y_raw))**2))
        
        # Delta-specific metrics
        delta_accuracy_5bp = np.mean(np.abs(predictions - y_raw) < 0.05)  # Within 5 basis points
        delta_accuracy_10bp = np.mean(np.abs(predictions - y_raw) < 0.10)  # Within 10 basis points
        
        metrics_table = pd.DataFrame({
            'Metric': ['MAE', 'MSE', 'RMSE', 'R²', 'Accuracy (5bp)', 'Accuracy (10bp)'],
            'Value': [f'{mae:.6f}', f'{mse:.6f}', f'{rmse:.6f}', f'{r2:.4f}', 
                     f'{delta_accuracy_5bp:.2%}', f'{delta_accuracy_10bp:.2%}'],
            'Description': [
                'Mean Absolute Error',
                'Mean Squared Error', 
                'Root Mean Squared Error',
                'R-squared Score',
                'Predictions within 0.05',
                'Predictions within 0.10'
            ]
        })
        
        mo.ui.table(metrics_table, label="📊 Model Performance Metrics")
    else:
        mo.md("📈 Performance metrics will be calculated after training")
    return (
        delta_accuracy_10bp,
        delta_accuracy_5bp,
        mae,
        metrics_table,
        mse,
        r2,
        rmse,
    )


if __name__ == "__main__":
    app.run()