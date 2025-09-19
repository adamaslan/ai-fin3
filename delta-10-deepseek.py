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
    # 🎯 Price Movement Probability Predictor
    
    Transformer-based model predicting probability of price reaching targets within time horizons.
    
    **Target**: Probability of 0-50% price movement up/down within 7-56 days
    **Price Targets**: Current ± $25 (25 levels each direction)
    **Time Horizons**: 7, 14, 28, 42, 56 days from Friday
    """)
    return

@app.cell
def __():
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
    from datetime import datetime, timedelta
    import warnings
    import math
    warnings.filterwarnings('ignore')
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        PLOTLY_AVAILABLE = True
    except ImportError:
        import matplotlib.pyplot as plt
        PLOTLY_AVAILABLE = False
    
    return (
        F, PLOTLY_AVAILABLE, RobustScaler, StandardScaler, accuracy_score,
        datetime, go, log_loss, make_subplots, math, nn, np, pd, plt, 
        px, roc_auc_score, timedelta, torch, warnings,
    )

@app.cell
def __(mo):
    # Transformer Architecture Controls
    d_model = mo.ui.slider(start=32, stop=512, step=32, value=128, label="Model Dimension")
    nhead = mo.ui.slider(start=1, stop=16, step=1, value=8, label="Attention Heads")
    num_layers = mo.ui.slider(start=1, stop=12, step=1, value=4, label="Transformer Layers")
    dropout_rate = mo.ui.slider(start=0.0, stop=0.5, step=0.05, value=0.2, label="Dropout Rate")
    
    # Training Parameters
    learning_rate = mo.ui.slider(start=0.0001, stop=0.01, step=0.0001, value=0.001, label="Learning Rate")
    batch_size = mo.ui.slider(start=16, stop=256, step=16, value=64, label="Batch Size")
    epochs = mo.ui.slider(start=10, stop=100, step=10, value=50, label="Epochs")
    
    mo.hstack([
        mo.vstack([mo.md("**Architecture**"), d_model, nhead, num_layers, dropout_rate]),
        mo.vstack([mo.md("**Training**"), learning_rate, batch_size, epochs])
    ])
    return batch_size, d_model, dropout_rate, epochs, learning_rate, nhead, num_layers

@app.cell
def __(np, pd):
    def generate_price_movement_data(n_samples=5000, seed=42):
        """Generate synthetic price movement probability data"""
        np.random.seed(seed)
        
        # Current price baseline
        current_price = 150.0
        
        # Price targets: ±$25 in $1 increments (50 targets total)
        price_targets = np.concatenate([
            np.arange(current_price - 25, current_price, 1),  # Below current
            np.arange(current_price + 1, current_price + 26, 1)  # Above current
        ])
        
        # Time horizons in days
        time_horizons = [7, 14, 28, 42, 56]
        
        data = []
        for _ in range(n_samples):
            # Random target and time selection
            target_price = np.random.choice(price_targets)
            time_horizon = np.random.choice(time_horizons)
            
            # Market features
            current_vol = np.random.uniform(0.15, 0.45)  # Implied volatility
            volume = np.random.lognormal(10, 1)  # Trading volume
            rsi = np.random.uniform(20, 80)  # RSI indicator
            macd = np.random.normal(0, 2)  # MACD
            bollinger_pos = np.random.uniform(0, 1)  # Position in Bollinger Bands
            vix = np.random.uniform(15, 35)  # VIX level
            
            # Calculate features
            price_distance = target_price - current_price
            price_pct_move = price_distance / current_price
            log_price_ratio = np.log(target_price / current_price)
            time_factor = np.sqrt(time_horizon / 365)
            
            # Volatility-adjusted metrics
            vol_adjusted_distance = abs(price_pct_move) / current_vol
            time_vol_interaction = time_factor * current_vol
            
            # Market regime indicators
            momentum_score = (rsi - 50) / 50  # Normalized momentum
            volatility_regime = 1 if current_vol > 0.25 else 0
            high_volume_regime = 1 if volume > np.exp(11) else 0
            
            # Calculate probability using realistic financial model
            # Higher probability for smaller moves, shorter timeframes, higher vol
            base_prob = 0.5
            
            # Distance penalty (larger moves less likely)
            distance_penalty = np.exp(-2 * abs(price_pct_move))
            
            # Time benefit (more time = higher probability)
            time_benefit = 1 - np.exp(-time_horizon / 14)
            
            # Volatility benefit (higher vol = higher probability of large moves)
            vol_benefit = current_vol / 0.3
            
            # Market conditions
            market_bias = 0.1 * momentum_score  # Momentum bias
            regime_adjustment = 0.1 if volatility_regime else 0
            
            # Final probability calculation
            probability = base_prob * distance_penalty * time_benefit * vol_benefit + market_bias + regime_adjustment
            probability = np.clip(probability, 0.01, 0.99)  # Bound between 1-99%
            
            # Convert to 0-50% range as specified
            probability_scaled = probability * 0.5  # Scale to 0-50%
            
            data.append({
                'current_price': current_price,
                'target_price': target_price,
                'price_distance': price_distance,
                'price_pct_move': price_pct_move,
                'log_price_ratio': log_price_ratio,
                'time_horizon': time_horizon,
                'time_factor': time_factor,
                'implied_vol': current_vol,
                'volume': volume,
                'rsi': rsi,
                'macd': macd,
                'bollinger_pos': bollinger_pos,
                'vix': vix,
                'vol_adjusted_distance': vol_adjusted_distance,
                'time_vol_interaction': time_vol_interaction,
                'momentum_score': momentum_score,
                'volatility_regime': volatility_regime,
                'high_volume_regime': high_volume_regime,
                'movement_probability': probability_scaled
            })
        
        return pd.DataFrame(data)
    
    # Generate dataset
    price_df = generate_price_movement_data()
    
    # Feature columns
    feature_columns = [
        'price_pct_move', 'log_price_ratio', 'time_factor', 'implied_vol',
        'volume', 'rsi', 'macd', 'bollinger_pos', 'vix',
        'vol_adjusted_distance', 'time_vol_interaction', 'momentum_score',
        'volatility_regime', 'high_volume_regime'
    ]
    
    X_data = price_df[feature_columns].values
    y_data = price_df['movement_probability'].values
    
    # Data summary
    mo.md(f"""
    ### Data Summary
    - **Samples**: {len(price_df):,}
    - **Features**: {len(feature_columns)}
    - **Probability Range**: {y_data.min():.1%} - {y_data.max():.1%}
    - **Price Targets**: ${price_df['target_price'].min():.0f} - ${price_df['target_price'].max():.0f}
    - **Time Horizons**: {sorted(price_df['time_horizon'].unique())} days
    """)
    return X_data, feature_columns, price_df, y_data

@app.cell
def __(F, nn, math, torch):
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=100):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))
        
        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class PriceProbabilityTransformer(nn.Module):
        def __init__(self, input_size, d_model=128, nhead=8, num_layers=4, dropout=0.2):
            super().__init__()
            
            # Input projection
            self.input_projection = nn.Sequential(
                nn.Linear(input_size, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            
            # Positional encoding
            self.pos_encoding = PositionalEncoding(d_model)
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            
            # Attention pooling
            self.attention_pool = nn.Sequential(
                nn.Linear(d_model, d_model//2),
                nn.GELU(),
                nn.Linear(d_model//2, 1),
                nn.Softmax(dim=1)
            )
            
            # Output layers
            self.output_layers = nn.Sequential(
                nn.Linear(d_model, d_model//2),
                nn.LayerNorm(d_model//2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model//2, d_model//4),
                nn.LayerNorm(d_model//4),
                nn.GELU(),
                nn.Linear(d_model//4, 1),
                nn.Sigmoid()  # Output 0-1, will scale to 0-50%
            )
            
            self.apply(self._init_weights)
        
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        
        def forward(self, x):
            # Project and add sequence dimension
            x_proj = self.input_projection(x).unsqueeze(1)
            
            # Add positional encoding
            x_pos = self.pos_encoding(x_proj)
            
            # Transformer processing
            transformer_out = self.transformer(x_pos)
            
            # Attention pooling
            attn_weights = self.attention_pool(transformer_out)
            pooled = torch.sum(attn_weights * transformer_out, dim=1)
            
            # Final prediction (0-1, represents 0-100% probability)
            prob = self.output_layers(pooled)
            
            # Scale to 0-50% as specified
            return prob * 0.5
    
    return PositionalEncoding, PriceProbabilityTransformer

@app.cell
def __(mo):
    train_button = mo.ui.button(label="Train Model", kind="success")
    mo.hstack([train_button, mo.md("Click to start training the probability predictor")])
    return train_button,

@app.cell
def __(
    PriceProbabilityTransformer, RobustScaler, X_data, batch_size, d_model,
    dropout_rate, epochs, learning_rate, log_loss, mo, nhead, nn, np,
    num_layers, torch, train_button, y_data,
):
    # Initialize training_results as a marimo state variable
    training_results = mo.state(None)
    
    # Use marimo's reactive execution - this will run when the button is clicked
    if train_button.value:
        # Prepare data
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_data)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_data).unsqueeze(1)
        
        # Train/validation split
        split_idx = int(0.8 * len(X_tensor))
        indices = torch.randperm(len(X_tensor))
        
        X_train = X_tensor[indices[:split_idx]]
        y_train = y_tensor[indices[:split_idx]]
        X_val = X_tensor[indices[split_idx:]]
        y_val = y_tensor[indices[split_idx:]]
        
        # Initialize model
        model = PriceProbabilityTransformer(
            input_size=X_data.shape[1],
            d_model=d_model.value,
            nhead=nhead.value,
            num_layers=num_layers.value,
            dropout=dropout_rate.value
        )
        
        # Optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate.value, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs.value)
        
        # Custom loss combining MSE and cross-entropy for probability
        def probability_loss(pred, target):
            # MSE for probability values
            mse = nn.MSELoss()(pred, target)
            
            # Add regularization to keep probabilities in reasonable range
            range_penalty = torch.mean(torch.clamp(pred - 0.5, min=0) ** 2)  # Penalty for >50%
            
            return mse + 0.1 * range_penalty
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(min(epochs.value, 50)):  # Limit for demo
            # Training
            model.train()
            epoch_loss = 0
            
            for i in range(0, len(X_train), batch_size.value):
                batch_X = X_train[i:i+batch_size.value]
                batch_y = y_train[i:i+batch_size.value]
                
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = probability_loss(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i in range(0, len(X_val), batch_size.value):
                    batch_X = X_val[i:i+batch_size.value]
                    batch_y = y_val[i:i+batch_size.value]
                    pred = model(batch_X)
                    loss = probability_loss(pred, batch_y)
                    val_loss += loss.item()
            
            scheduler.step()
            
            avg_train_loss = epoch_loss / (len(X_train) // batch_size.value)
            avg_val_loss = val_loss / (len(X_val) // batch_size.value)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
        
        # Final predictions
        model.eval()
        with torch.no_grad():
            all_predictions = model(X_tensor).numpy().flatten()
        
        # Update the state with training results
        training_results.value = {
            'model': model,
            'scaler': scaler,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'predictions': all_predictions,
            'epochs_trained': len(train_losses)
        }
    
    # Display training status
    if training_results.value:
        # Calculate metrics
        mae = np.mean(np.abs(training_results.value['predictions'] - y_data))
        rmse = np.sqrt(np.mean((training_results.value['predictions'] - y_data) ** 2))
        
        mo.md(f"""
        Training Complete!
        - **Epochs**: {len(training_results.value['train_losses'])}
        - **MAE**: {mae:.4f}
        - **RMSE**: {rmse:.4f}
        - **Parameters**: {sum(p.numel() for p in training_results.value['model'].parameters()):,}
        """)
    else:
        mo.md("Click **Train Model** to start training")
    
    return training_results,

@app.cell
def __(PLOTLY_AVAILABLE, go, mo, training_results):
    if training_results.value and PLOTLY_AVAILABLE:
        # Training visualization
        fig = go.Figure()
        epochs_range = list(range(1, len(training_results.value['train_losses']) + 1))
        
        fig.add_trace(go.Scatter(
            x=epochs_range, y=training_results.value['train_losses'], 
            name='Train Loss', line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=epochs_range, y=training_results.value['val_losses'], 
            name='Val Loss', line=dict(color='red')
        ))
        
        fig.update_layout(
            title="Training Progress",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=400
        )
        
        mo.ui.plotly(fig)
    elif training_results.value:
        mo.md("Training completed! (Plotly visualization not available)")
    else:
        mo.md("Training visualization will appear after model training")
    return epochs_range, fig

@app.cell
def __(mo, training_results):
    if training_results.value:
        # Interactive prediction interface
        pred_target = mo.ui.slider(
            start=125, stop=175, step=1, value=160,
            label="Target Price ($)"
        )
        
        pred_days = mo.ui.dropdown(
            options=[7, 14, 28, 42, 56],
            value=14,
            label="Time Horizon (days)"
        )
        
        pred_vol = mo.ui.slider(
            start=15, stop=45, step=1, value=25,
            label="Implied Volatility (%)"
        )
        
        pred_rsi = mo.ui.slider(
            start=20, stop=80, step=1, value=50,
            label="RSI"
        )
        
        mo.hstack([
            mo.vstack([pred_target, pred_days]),
            mo.vstack([pred_vol, pred_rsi])
        ])
    else:
        pred_target = pred_days = pred_vol = pred_rsi = None
        mo.md("Complete training to access live predictions")
    
    return pred_days, pred_rsi, pred_target, pred_vol

@app.cell
def __(
    PLOTLY_AVAILABLE, go, mo, np, pred_days, pred_rsi, pred_target, pred_vol,
    torch, training_results,
):
    if training_results.value and all(x is not None for x in [pred_target, pred_days, pred_vol, pred_rsi]):
        
        # Current market parameters
        current_price = 150.0
        
        # Extract prediction parameters
        target_price = pred_target.value
        time_horizon = pred_days.value
        vol_input = pred_vol.value / 100
        rsi_input = pred_rsi.value
        
        # Calculate features
        price_distance = target_price - current_price
        price_pct_move = price_distance / current_price
        log_price_ratio = np.log(target_price / current_price)
        time_factor = np.sqrt(time_horizon / 365)
        
        # Mock additional features for demo
        volume = np.exp(10.5)  # Typical volume
        macd = 0.0  # Neutral MACD
        bollinger_pos = 0.5  # Mid Bollinger
        vix = 20.0  # Normal VIX
        
        vol_adjusted_distance = abs(price_pct_move) / vol_input
        time_vol_interaction = time_factor * vol_input
        momentum_score = (rsi_input - 50) / 50
        volatility_regime = 1 if vol_input > 0.25 else 0
        high_volume_regime = 1 if volume > np.exp(11) else 0
        
        # Create feature vector
        live_features = np.array([[
            price_pct_move, log_price_ratio, time_factor, vol_input,
            volume, rsi_input, macd, bollinger_pos, vix,
            vol_adjusted_distance, time_vol_interaction, momentum_score,
            volatility_regime, high_volume_regime
        ]])
        
        # Scale and predict
        live_features_scaled = training_results.value['scaler'].transform(live_features)
        
        with torch.no_grad():
            training_results.value['model'].eval()
            prediction = training_results.value['model'](torch.FloatTensor(live_features_scaled))
            probability_pct = prediction.item() * 100  # Convert to percentage
        
        # Analysis
        direction = "UP" if target_price > current_price else "DOWN"
        move_size = abs(price_pct_move) * 100
        days_text = f"{time_horizon} days"
        
        if PLOTLY_AVAILABLE:
            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability_pct,
                title = {'text': f"Probability of reaching ${target_price}"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 50]},  # 0-50% range as specified
                    'bar': {'color': "green" if direction == "UP" else "red"},
                    'steps': [
                        {'range': [0, 10], 'color': "lightgray"},
                        {'range': [10, 25], 'color': "yellow"},
                        {'range': [25, 40], 'color': "orange"},
                        {'range': [40, 50], 'color': "darkgreen"}
                    ],
                }
            ))
            fig_gauge.update_layout(height=400)
        
        # Detailed results
        results = mo.md(f"""
        ### Prediction Results
        
        **Movement Analysis:**
        - **Direction**: {direction} by ${abs(price_distance):.0f} ({move_size:.1f}%)
        - **Time Horizon**: {days_text}
        - **Probability**: {probability_pct:.1f}%
        
        **Market Context:**
        - **Current Price**: ${current_price}
        - **Target Price**: ${target_price}
        - **Implied Vol**: {vol_input:.1%}
        - **RSI**: {rsi_input} ({"Overbought" if rsi_input > 70 else "Oversold" if rsi_input < 30 else "Neutral"})
        - **Vol Regime**: {"High" if volatility_regime else "Normal"}
        
        **Risk Assessment:**
        - **Difficulty**: {"High" if move_size > 10 else "Medium" if move_size > 5 else "Low"}
        - **Time Pressure**: {"High" if time_horizon <= 7 else "Medium" if time_horizon <= 28 else "Low"}
        - **Confidence**: {"High" if probability_pct > 30 else "Medium" if probability_pct > 15 else "Low"}
        """)
        
        if PLOTLY_AVAILABLE:
            mo.hstack([mo.ui.plotly(fig_gauge), results])
        else:
            results
    else:
        mo.md("Set prediction parameters to get probability forecast")
    
    return (
        bollinger_pos, current_price, direction, fig_gauge, live_features,
        live_features_scaled, log_price_ratio, macd, momentum_score,
        move_size, prediction, price_distance, price_pct_move,
        probability_pct, results, rsi_input, target_price, time_factor,
        time_horizon, time_vol_interaction, vol_adjusted_distance,
        vol_input, volatility_regime, volume, vix,
    )

@app.cell
def __(PLOTLY_AVAILABLE, go, mo, np, training_results, y_data):
    if training_results.value and PLOTLY_AVAILABLE:
        # Performance analysis
        predictions = training_results.value['predictions']
        actuals = y_data
        
        fig_performance = go.Figure()
        
        # Perfect prediction line
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        fig_performance.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name='Perfect Prediction',
                      line=dict(dash='dash', color='gray'))
        )
        
        # Actual vs predicted
        fig_performance.add_trace(
            go.Scatter(x=actuals, y=predictions, mode='markers',
                      name='Predictions', opacity=0.6,
                      marker=dict(color='blue', size=4))
        )
        
        fig_performance.update_layout(
            title="Model Performance: Actual vs Predicted Probabilities",
            xaxis_title="Actual Probability",
            yaxis_title="Predicted Probability",
            height=400
        )
        
        # Error distribution
        errors = predictions - actuals
        fig_error = go.Figure()
        fig_error.add_trace(go.Histogram(x=errors, nbinsx=50, name='Prediction Errors'))
        fig_error.update_layout(
            title="Prediction Error Distribution",
            xaxis_title="Prediction Error",
            yaxis_title="Frequency",
            height=300
        )
        
        mo.vstack([mo.ui.plotly(fig_performance), mo.ui.plotly(fig_error)])
    elif training_results.value:
        mo.md("Performance analysis completed! (Plotly not available)")
    else:
        mo.md("Performance analysis will appear after training")
    
    return actuals, errors, fig_error, fig_performance, max_val, min_val, predictions

if __name__ == "__main__":
    app.run()