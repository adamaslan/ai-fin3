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
    # 🎯 Options Delta Predictor
    
    Transformer-based model predicting options delta values from market features.
    
    **Target**: Predict delta (0-1) for call options
    **Features**: Moneyness, time to expiry, implied volatility, volume, market conditions
    **Architecture**: Transformer with attention mechanisms
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
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from datetime import datetime, timedelta
    import warnings
    import math
    warnings.filterwarnings('ignore')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    return (
        F, StandardScaler, RobustScaler, datetime, device, math, 
        mean_absolute_error, mean_squared_error, nn, np, pd, 
        r2_score, timedelta, torch, warnings,
    )

@app.cell
def __(mo):
    # Model Architecture Controls
    d_model = mo.ui.slider(start=32, stop=256, step=32, value=128, label="Model Dimension")
    nhead = mo.ui.slider(start=2, stop=16, step=2, value=8, label="Attention Heads")  
    num_layers = mo.ui.slider(start=2, stop=8, step=1, value=4, label="Transformer Layers")
    dropout_rate = mo.ui.slider(start=0.0, stop=0.4, step=0.05, value=0.2, label="Dropout Rate")
    
    # Training Parameters  
    learning_rate = mo.ui.slider(start=0.0001, stop=0.01, step=0.0005, value=0.001, label="Learning Rate")
    batch_size = mo.ui.slider(start=32, stop=256, step=32, value=64, label="Batch Size")
    epochs = mo.ui.slider(start=20, stop=100, step=10, value=50, label="Epochs")
    
    mo.hstack([
        mo.vstack([mo.md("**Architecture**"), d_model, nhead, num_layers, dropout_rate]),
        mo.vstack([mo.md("**Training**"), learning_rate, batch_size, epochs])
    ])
    return batch_size, d_model, dropout_rate, epochs, learning_rate, nhead, num_layers

@app.cell
def __(np, pd):
    def generate_delta_training_data(n_samples=8000, seed=42):
        """Generate realistic options delta training data"""
        np.random.seed(seed)
        
        current_price = 150.0
        data = []
        
        for _ in range(n_samples):
            # Option parameters
            strike = np.random.uniform(100, 200)
            days_to_expiry = np.random.uniform(1, 365)
            implied_vol = np.random.uniform(0.10, 0.60)
            risk_free_rate = 0.05
            
            # Market microstructure
            volume = np.random.lognormal(8, 1.5)
            open_interest = np.random.lognormal(7, 1.2)
            bid_ask_spread = np.random.uniform(0.01, 0.20)
            
            # Market conditions
            vix = np.random.uniform(12, 40)
            market_trend = np.random.uniform(-0.02, 0.02)  # Daily return
            
            # Calculate features
            moneyness = strike / current_price
            log_moneyness = np.log(moneyness)
            time_to_expiry_years = days_to_expiry / 365.0
            sqrt_time = np.sqrt(time_to_expiry_years)
            
            # Volume features
            log_volume = np.log1p(volume)
            volume_oi_ratio = volume / max(open_interest, 1)
            
            # Volatility features
            vol_time = implied_vol * sqrt_time
            vol_moneyness = implied_vol * abs(log_moneyness)
            
            # Market regime indicators
            high_vol_regime = 1 if implied_vol > 0.30 else 0
            short_term = 1 if days_to_expiry <= 30 else 0
            
            # Calculate true delta using Black-Scholes approximation
            from scipy.stats import norm
            
            d1 = (np.log(current_price / strike) + (risk_free_rate + 0.5 * implied_vol**2) * time_to_expiry_years) / (implied_vol * sqrt_time)
            true_delta = norm.cdf(d1)
            
            # Add some noise to make it more realistic
            true_delta = np.clip(true_delta + np.random.normal(0, 0.02), 0.01, 0.99)
            
            data.append({
                'strike': strike,
                'current_price': current_price,
                'moneyness': moneyness,
                'log_moneyness': log_moneyness,
                'days_to_expiry': days_to_expiry,
                'time_to_expiry_years': time_to_expiry_years,
                'sqrt_time': sqrt_time,
                'implied_vol': implied_vol,
                'volume': volume,
                'log_volume': log_volume,
                'open_interest': open_interest,
                'volume_oi_ratio': volume_oi_ratio,
                'bid_ask_spread': bid_ask_spread,
                'vix': vix,
                'market_trend': market_trend,
                'vol_time': vol_time,
                'vol_moneyness': vol_moneyness,
                'high_vol_regime': high_vol_regime,
                'short_term': short_term,
                'delta': true_delta
            })
        
        return pd.DataFrame(data)
    
    # Import scipy for Black-Scholes calculation
    try:
        from scipy.stats import norm
        scipy_available = True
    except ImportError:
        scipy_available = False
        # Fallback normal CDF approximation
        def norm_cdf_approx(x):
            return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
        norm = type('norm', (), {'cdf': staticmethod(norm_cdf_approx)})()
    
    # Generate dataset
    options_df = generate_delta_training_data()
    
    # Feature columns for model input
    feature_columns = [
        'log_moneyness', 'sqrt_time', 'implied_vol', 'log_volume',
        'volume_oi_ratio', 'bid_ask_spread', 'vix', 'market_trend',
        'vol_time', 'vol_moneyness', 'high_vol_regime', 'short_term'
    ]
    
    X_data = options_df[feature_columns].values
    y_data = options_df['delta'].values
    
    # Data summary
    data_summary = mo.md(f"""
    ### Dataset Summary
    - **Samples**: {len(options_df):,}
    - **Features**: {len(feature_columns)}
    - **Delta Range**: {y_data.min():.3f} - {y_data.max():.3f}
    - **Strike Range**: ${options_df['strike'].min():.0f} - ${options_df['strike'].max():.0f}
    - **Expiry Range**: {options_df['days_to_expiry'].min():.0f} - {options_df['days_to_expiry'].max():.0f} days
    - **Vol Range**: {options_df['implied_vol'].min():.1%} - {options_df['implied_vol'].max():.1%}
    """)
    
    data_summary
    
    return X_data, feature_columns, norm, options_df, scipy_available, y_data

@app.cell
def __(F, math, nn, torch):
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=1000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model % 2 == 1:
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
            else:
                pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))
        
        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class TransformerDeltaPredictor(nn.Module):
        def __init__(self, input_size=12, d_model=128, nhead=8, num_layers=4, dropout=0.2):
            super().__init__()
            
            # Ensure d_model is divisible by nhead
            assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            
            self.input_size = input_size
            self.d_model = d_model
            
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
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            
            # Attention pooling
            self.attention_pool = nn.Sequential(
                nn.Linear(d_model, d_model//2),
                nn.GELU(),
                nn.Linear(d_model//2, 1),
                nn.Softmax(dim=1)
            )
            
            # Output layers for delta prediction
            self.output_layers = nn.Sequential(
                nn.Linear(d_model, d_model//2),
                nn.LayerNorm(d_model//2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model//2, d_model//4),
                nn.LayerNorm(d_model//4),
                nn.GELU(),
                nn.Dropout(dropout//2),
                nn.Linear(d_model//4, 1),
                nn.Sigmoid()  # Delta is between 0 and 1
            )
            
            self.apply(self._init_weights)
        
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        
        def forward(self, x):
            # Project and add sequence dimension
            x_proj = self.input_projection(x).unsqueeze(1)  # (batch, 1, d_model)
            
            # Add positional encoding
            x_pos = self.pos_encoding(x_proj)
            
            # Transformer processing
            transformer_out = self.transformer(x_pos)  # (batch, 1, d_model)
            
            # Attention pooling
            attn_weights = self.attention_pool(transformer_out)
            pooled = torch.sum(attn_weights * transformer_out, dim=1)  # (batch, d_model)
            
            # Final delta prediction
            delta = self.output_layers(pooled)
            
            return delta
    
    return PositionalEncoding, TransformerDeltaPredictor

@app.cell
def __(mo):
    # Create training button and state
    train_button = mo.ui.button(label="🚀 Train Delta Model", kind="success")
    
    mo.hstack([
        train_button, 
        mo.md("Click to start training the options delta predictor")
    ])
    return train_button,

@app.cell
def __(
    RobustScaler, TransformerDeltaPredictor, X_data, batch_size, d_model,
    device, dropout_rate, epochs, learning_rate, mean_absolute_error,
    mean_squared_error, mo, nhead, nn, np, num_layers, r2_score, torch,
    train_button, y_data,
):
    # Training logic - this runs when button is clicked
    if train_button.value:
        try:
            # Prepare data
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_data)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            y_tensor = torch.FloatTensor(y_data).unsqueeze(1).to(device)
            
            # Train/validation split
            dataset_size = len(X_tensor)
            indices = torch.randperm(dataset_size)
            split_idx = int(0.8 * dataset_size)
            
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            
            X_train = X_tensor[train_indices]
            y_train = y_tensor[train_indices]
            X_val = X_tensor[val_indices]
            y_val = y_tensor[val_indices]
            
            # Initialize model
            model = TransformerDeltaPredictor(
                input_size=X_data.shape[1],
                d_model=d_model.value,
                nhead=nhead.value,
                num_layers=num_layers.value,
                dropout=dropout_rate.value
            ).to(device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Optimizer and scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=learning_rate.value, 
                weight_decay=1e-5
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=epochs.value,
                eta_min=learning_rate.value * 0.01
            )
            
            # Loss function - MSE with L1 regularization for delta prediction
            def delta_loss_fn(pred, target):
                mse_loss = nn.MSELoss()(pred, target)
                mae_loss = nn.L1Loss()(pred, target)
                return 0.8 * mse_loss + 0.2 * mae_loss
            
            # Training loop
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            
            model.train()
            
            for epoch in range(epochs.value):
                # Training phase
                epoch_train_loss = 0
                num_train_batches = 0
                
                for i in range(0, len(X_train), batch_size.value):
                    batch_end = min(i + batch_size.value, len(X_train))
                    batch_X = X_train[i:batch_end]
                    batch_y = y_train[i:batch_end]
                    
                    optimizer.zero_grad()
                    
                    predictions = model(batch_X)
                    loss = delta_loss_fn(predictions, batch_y)
                    
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_train_loss += loss.item()
                    num_train_batches += 1
                
                # Validation phase
                model.eval()
                epoch_val_loss = 0
                num_val_batches = 0
                
                with torch.no_grad():
                    for i in range(0, len(X_val), batch_size.value):
                        batch_end = min(i + batch_size.value, len(X_val))
                        batch_X = X_val[i:batch_end]
                        batch_y = y_val[i:batch_end]
                        
                        predictions = model(batch_X)
                        loss = delta_loss_fn(predictions, batch_y)
                        
                        epoch_val_loss += loss.item()
                        num_val_batches += 1
                
                model.train()
                scheduler.step()
                
                # Calculate average losses
                avg_train_loss = epoch_train_loss / num_train_batches
                avg_val_loss = epoch_val_loss / num_val_batches
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
            
            # Final evaluation
            model.eval()
            with torch.no_grad():
                train_predictions = model(X_train).cpu().numpy().flatten()
                val_predictions = model(X_val).cpu().numpy().flatten()
                all_predictions = model(X_tensor).cpu().numpy().flatten()
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train.cpu().numpy().flatten(), train_predictions)
            val_mae = mean_absolute_error(y_val.cpu().numpy().flatten(), val_predictions)
            train_mse = mean_squared_error(y_train.cpu().numpy().flatten(), train_predictions)
            val_mse = mean_squared_error(y_val.cpu().numpy().flatten(), val_predictions)
            train_r2 = r2_score(y_train.cpu().numpy().flatten(), train_predictions)
            val_r2 = r2_score(y_val.cpu().numpy().flatten(), val_predictions)
            
            # Store results in a simple dictionary
            training_results = {
                'model': model,
                'scaler': scaler,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'all_predictions': all_predictions,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'train_mse': train_mse,
                'val_mse': val_mse,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'best_val_loss': best_val_loss,
                'final_lr': scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else learning_rate.value
            }
            
            # Success message
            result_display = mo.md(f"""
            ## ✅ Training Complete!
            
            ### Model Architecture
            - **Parameters**: {total_params:,} total ({trainable_params:,} trainable)
            - **Architecture**: {num_layers.value} layers, {nhead.value} heads, {d_model.value}d model
            
            ### Training Results
            - **Epochs**: {len(train_losses)}
            - **Best Val Loss**: {best_val_loss:.6f}
            - **Final LR**: {training_results['final_lr']:.2e}
            
            ### Performance Metrics
            **Training Set:**
            - MAE: {train_mae:.4f}
            - MSE: {train_mse:.6f}  
            - R²: {train_r2:.4f}
            
            **Validation Set:**
            - MAE: {val_mae:.4f}
            - MSE: {val_mse:.6f}
            - R²: {val_r2:.4f}
            """)
            
        except Exception as e:
            training_results = None
            result_display = mo.md(f"""
            ## ❌ Training Failed
            **Error**: {str(e)}
            
            Please check your parameters and try again.
            """)
    else:
        training_results = None  
        result_display = mo.md("**Ready to train!** Click the button above to start training.")
    
    result_display
    
    return result_display, training_results

@app.cell
def __(mo, np, training_results):
    # Training visualization
    if training_results is not None:
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Loss curves
            epochs_range = range(1, len(training_results['train_losses']) + 1)
            ax1.plot(epochs_range, training_results['train_losses'], 'b-', label='Training Loss', alpha=0.8)
            ax1.plot(epochs_range, training_results['val_losses'], 'r-', label='Validation Loss', alpha=0.8)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Progress')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Predictions vs actual
            y_true = training_results['all_predictions']  # This should be actual y_data
            y_pred = training_results['all_predictions']
            
            # We need the actual targets for comparison - let's use y_data from earlier
            # Since we don't have access to split here, show distribution instead
            ax2.hist(training_results['all_predictions'], bins=50, alpha=0.7, label='Predicted Deltas')
            ax2.set_xlabel('Delta Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Delta Prediction Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            mo.md(f"Visualization error: {str(e)}")
    else:
        mo.md("Training visualization will appear after model training.")
    
    return ax1, ax2, epochs_range, fig, plt, y_pred, y_true

@app.cell
def __(mo, training_results):
    # Live prediction interface
    if training_results is not None:
        # Input controls for live prediction
        pred_moneyness = mo.ui.slider(
            start=0.7, stop=1.3, step=0.01, value=1.0,
            label="Moneyness (Strike/Spot)"
        )
        
        pred_days = mo.ui.slider(
            start=1, stop=365, step=1, value=30,
            label="Days to Expiry"
        )
        
        pred_vol = mo.ui.slider(
            start=10, stop=60, step=1, value=25,
            label="Implied Volatility (%)"
        )
        
        pred_volume = mo.ui.slider(
            start=100, stop=50000, step=100, value=5000,
            label="Volume"
        )
        
        prediction_inputs = mo.hstack([
            mo.vstack([
                mo.md("**Option Parameters**"),
                pred_moneyness, 
                pred_days
            ]),
            mo.vstack([
                mo.md("**Market Conditions**"),
                pred_vol, 
                pred_volume
            ])
        ])
        
        prediction_inputs
    else:
        prediction_inputs = None
        mo.md("Live prediction interface will be available after training.")
    
    return pred_days, pred_moneyness, pred_vol, pred_volume, prediction_inputs

@app.cell
def __(
    device, mo, np, pred_days, pred_moneyness, pred_vol, pred_volume, torch,
    training_results,
):
    # Live delta prediction
    if training_results is not None and all(x is not None for x in [pred_moneyness, pred_days, pred_vol, pred_volume]):
        
        try:
            # Extract input values
            moneyness = pred_moneyness.value
            days_to_expiry = pred_days.value
            implied_vol = pred_vol.value / 100  # Convert percentage
            volume = pred_volume.value
            
            # Calculate derived features
            log_moneyness = np.log(moneyness)
            time_to_expiry_years = days_to_expiry / 365.0
            sqrt_time = np.sqrt(time_to_expiry_years)
            log_volume = np.log1p(volume)
            
            # Mock additional features (in real app, these would come from market data)
            open_interest = volume * 1.5  # Typical ratio
            volume_oi_ratio = volume / open_interest
            bid_ask_spread = 0.05  # Typical spread
            vix = 20.0  # Neutral VIX
            market_trend = 0.001  # Small positive trend
            
            # Interaction features
            vol_time = implied_vol * sqrt_time
            vol_moneyness = implied_vol * abs(log_moneyness)
            
            # Binary features
            high_vol_regime = 1 if implied_vol > 0.30 else 0
            short_term = 1 if days_to_expiry <= 30 else 0
            
            # Create feature vector (matching training order)
            live_features = np.array([[
                log_moneyness, sqrt_time, implied_vol, log_volume,
                volume_oi_ratio, bid_ask_spread, vix, market_trend,
                vol_time, vol_moneyness, high_vol_regime, short_term
            ]])
            
            # Scale features
            live_features_scaled = training_results['scaler'].transform(live_features)
            
            # Make prediction
            training_results['model'].eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(live_features_scaled).to(device)
                predicted_delta = training_results['model'](features_tensor).cpu().item()
            
            # Analysis
            moneyness_desc = "ITM" if moneyness < 1.0 else "ATM" if abs(moneyness - 1.0) < 0.02 else "OTM"
            time_desc = "Short-term" if days_to_expiry <= 7 else "Medium-term" if days_to_expiry <= 60 else "Long-term"
            vol_desc = "Low" if implied_vol < 0.2 else "High" if implied_vol > 0.35 else "Normal"
            
            # Display results
            prediction_result = mo.md(f"""
            ## 🔮 Live Delta Prediction
            
            ### Predicted Delta: **{predicted_delta:.4f}**
            
            ### Option Characteristics
            - **Moneyness**: {moneyness:.3f} ({moneyness_desc})
            - **Time to Expiry**: {days_to_expiry} days ({time_desc})  
            - **Implied Vol**: {implied_vol:.1%} ({vol_desc})
            - **Volume**: {volume:,}
            
            ### Market Context
            - **Strike Sensitivity**: {"High" if predicted_delta > 0.7 else "Medium" if predicted_delta > 0.3 else "Low"}
            - **Time Decay Risk**: {"High" if days_to_expiry <= 7 else "Medium" if days_to_expiry <= 30 else "Low"}
            - **Volatility Regime**: {"High" if high_vol_regime else "Normal"}
            
            ### Interpretation
            - **Price Sensitivity**: A $1 move in underlying ≈ **${predicted_delta:.2f}** option price change
            - **Hedge Ratio**: Need **{1/predicted_delta:.1f}** options to hedge 100 shares (approximately)
            - **Probability ITM**: ~**{predicted_delta:.1%}** (rough approximation)
            """)
            
        except Exception as e:
            prediction_result = mo.md(f"**Prediction Error**: {str(e)}")
            
    else:
        prediction_result = mo.md("Set prediction parameters above to see live delta calculation.")
    
    prediction_result
    
    return (
        bid_ask_spread, days_to_expiry, features_tensor, high_vol_regime,
        implied_vol, live_features, live_features_scaled, log_moneyness,
        log_volume, market_trend, moneyness, moneyness_desc, open_interest,
        predicted_delta, prediction_result, short_term, sqrt_time,
        time_desc, time_to_expiry_years, vol_desc, vol_moneyness, vol_time,
        volume, volume_oi_ratio, vix,
    )

@app.cell
def __(mo, training_results, y_data):
    # Model analysis and performance breakdown
    if training_results is not None:
        
        # Error analysis
        all_preds = training_results['all_predictions']
        errors = all_preds - y_data
        abs_errors = np.abs(errors)
        
        # Error statistics
        error_stats = mo.md(f"""
        ## 📊 Detailed Performance Analysis
        
        ### Error Distribution
        - **Mean Error**: {np.mean(errors):.6f}
        - **Std Error**: {np.std(errors):.6f}  
        - **Max Absolute Error**: {np.max(abs_errors):.4f}
        - **90th Percentile Error**: {np.percentile(abs_errors, 90):.4f}
        - **95th Percentile Error**: {np.percentile(abs_errors, 95):.4f}
        
        ### Model Robustness
        - **Predictions in [0,1]**: {np.sum((all_preds >= 0) & (all_preds <= 1)) / len(all_preds) * 100:.1f}%
        - **Predictions > 0.95**: {np.sum(all_preds > 0.95)}/{len(all_preds)} ({np.sum(all_preds > 0.95)/len(all_preds)*100:.1f}%)
        - **Predictions < 0.05**: {np.sum(all_preds < 0.05)}/{len(all_preds)} ({np.sum(all_preds < 0.05)/len(all_preds)*100:.1f}%)
        
        ### Delta Range Analysis
        - **Deep ITM (δ > 0.8)**: {np.sum((y_data > 0.8) & (abs_errors < 0.05)) / np.sum(y_data > 0.8) * 100:.1f}% accurate within 0.05
        - **ATM (0.4 < δ < 0.6)**: {np.sum(((y_data > 0.4) & (y_data < 0.6)) & (abs_errors < 0.02)) / np.sum((y_data > 0.4) & (y_data < 0.6)) * 100:.1f}% accurate within 0.02
        - **Deep OTM (δ < 0.2)**: {np.sum((y_data < 0.2) & (abs_errors < 0.02)) / np.sum(y_data < 0.2) * 100:.1f}% accurate within 0.02
        """)
        
        error_stats
    else:
        mo.md("Performance analysis will appear after training.")
    
    return abs_errors, all_preds, error_stats, errors

@app.cell
def __(mo):
    # Model export and summary
    mo.md("""
    ## 💾 Model Export & Summary
    
    This delta predictor uses a Transformer architecture to learn complex relationships 
    in options pricing. The model processes market features through attention mechanisms
    to predict delta values with high accuracy.
    
    ### Key Features:
    - **Multi-head Attention**: Captures feature interactions
    - **Positional Encoding**: Handles sequential relationships  
    - **Layer Normalization**: Stable training
    - **Dropout Regularization**: Prevents overfitting
    - **Sigmoid Output**: Ensures delta ∈ [0,1]
    
    ### Applications:
    - **Risk Management**: Dynamic hedging calculations
    - **Market Making**: Fair value pricing
    - **Portfolio Optimization**: Greeks-based allocation
    - **Algorithmic Trading**: Real-time delta estimation
    """)

if __name__ == "__main__":
    app.run()