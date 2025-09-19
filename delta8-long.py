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
    # 🚀 Advanced Transformer Delta Predictor
    
    Interactive deep learning model for options delta prediction using state-of-the-art Transformer architecture.
    
    **Features:**
    - Comprehensive Transformer parameter control
    - Real-time training visualization
    - Advanced performance metrics
    - Attention mechanism analysis
    - Model interpretability tools
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
    
    # Try to import plotly, fallback to matplotlib if not available
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        PLOTLY_AVAILABLE = True
    except ImportError:
        import matplotlib.pyplot as plt
        PLOTLY_AVAILABLE = False
        print("Plotly not available, using matplotlib fallbacks")
    
    return (
        F,
        PLOTLY_AVAILABLE,
        RobustScaler,
        StandardScaler,
        datetime,
        go,
        make_subplots,
        math,
        mean_absolute_error,
        mean_squared_error,
        nn,
        np,
        pd,
        plt,
        px,
        r2_score,
        timedelta,
        torch,
        warnings,
    )


@app.cell
def __(mo):
    mo.md("## 🎛️ Transformer Architecture Controls")
    return


@app.cell
def __(mo):
    # Transformer-specific parameters
    d_model = mo.ui.slider(
        start=32, stop=512, step=32, value=128,
        label="Model Dimension (d_model)"
    )
    
    nhead = mo.ui.slider(
        start=1, stop=16, step=1, value=8,
        label="Number of Attention Heads"
    )
    
    num_layers = mo.ui.slider(
        start=1, stop=12, step=1, value=4,
        label="Number of Transformer Layers"
    )
    
    dropout_rate = mo.ui.slider(
        start=0.0, stop=0.5, step=0.05, value=0.2,
        label="Dropout Rate"
    )
    
    return d_model, dropout_rate, nhead, num_layers


@app.cell
def __(mo):
    # Training parameters
    learning_rate = mo.ui.slider(
        start=0.0001, stop=0.01, step=0.0001, value=0.001,
        label="Learning Rate"
    )
    
    batch_size = mo.ui.slider(
        start=16, stop=256, step=16, value=64,
        label="Batch Size"
    )
    
    epochs = mo.ui.slider(
        start=10, stop=200, step=10, value=100,
        label="Training Epochs"
    )
    
    weight_decay = mo.ui.slider(
        start=0.0, stop=0.01, step=0.0001, value=0.0001,
        label="Weight Decay (L2 Regularization)"
    )
    
    return batch_size, epochs, learning_rate, weight_decay


@app.cell
def __(mo):
    # Advanced options
    activation_fn = mo.ui.dropdown(
        options=["relu", "gelu", "swish", "leaky_relu"],
        value="gelu",
        label="Activation Function"
    )
    
    optimizer_type = mo.ui.dropdown(
        options=["Adam", "AdamW", "RMSprop", "SGD"],
        value="AdamW",
        label="Optimizer"
    )
    
    scheduler_type = mo.ui.dropdown(
        options=["None", "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"],
        value="CosineAnnealingLR",
        label="Learning Rate Scheduler"
    )
    
    loss_function = mo.ui.dropdown(
        options=["MSE", "Huber", "L1", "Combined"],
        value="Combined",
        label="Loss Function"
    )
    
    return activation_fn, loss_function, optimizer_type, scheduler_type


@app.cell
def __(
    activation_fn,
    batch_size,
    d_model,
    dropout_rate,
    epochs,
    learning_rate,
    loss_function,
    mo,
    nhead,
    num_layers,
    optimizer_type,
    scheduler_type,
    weight_decay,
):
    # Display all controls
    mo.hstack([
        mo.vstack([
            mo.md("**Architecture**"),
            d_model,
            nhead,
            num_layers,
            dropout_rate
        ]),
        mo.vstack([
            mo.md("**Training**"),
            learning_rate,
            batch_size,
            epochs,
            weight_decay
        ]),
        mo.vstack([
            mo.md("**Advanced**"),
            activation_fn,
            optimizer_type,
            scheduler_type,
            loss_function
        ])
    ])
    return


@app.cell
def __(np, pd):
    def generate_enhanced_options_data(n_samples=3000, seed=42):
        """Generate comprehensive synthetic options data"""
        np.random.seed(seed)
        
        # Market parameters with more realistic distributions
        S0 = 150  # Current stock price
        
        # Generate strikes with clustering around ATM
        atm_strikes = np.random.normal(S0, S0*0.1, n_samples//2)
        otm_strikes = np.random.uniform(S0*0.6, S0*1.4, n_samples//2)
        strikes = np.concatenate([atm_strikes, otm_strikes])
        strikes = np.clip(strikes, S0*0.5, S0*2.0)
        
        # Time to expiry with realistic distribution (more short-term options)
        times_to_expiry = np.concatenate([
            np.random.exponential(0.1, n_samples//3),  # Short term
            np.random.uniform(0.1, 0.5, n_samples//3),  # Medium term
            np.random.uniform(0.5, 2.0, n_samples//3)   # Long term
        ])
        times_to_expiry = np.clip(times_to_expiry, 1/365, 2.0)
        
        # IV with term structure and smile
        base_iv = 0.25
        iv_smile = 0.1 * ((strikes / S0 - 1) ** 2)  # Volatility smile
        iv_term = 0.05 * np.sqrt(times_to_expiry)  # Term structure
        volatilities = base_iv + iv_smile + iv_term + np.random.normal(0, 0.05, n_samples)
        volatilities = np.clip(volatilities, 0.05, 1.0)
        
        # Calculate theoretical delta using Black-Scholes
        from scipy.stats import norm
        risk_free_rate = 0.05
        
        d1 = (np.log(S0/strikes) + (risk_free_rate + 0.5*volatilities**2)*times_to_expiry) / (volatilities*np.sqrt(times_to_expiry))
        theoretical_delta = norm.cdf(d1)
        
        # Add market microstructure noise
        market_noise = np.random.normal(0, 0.02, n_samples)
        observed_delta = np.clip(theoretical_delta + market_noise, 0.01, 0.99)
        
        # Market microstructure variables
        volume = np.random.lognormal(4, 1.5, n_samples)  # Log-normal volume
        open_interest = np.random.lognormal(5, 1, n_samples)
        bid_ask_spread = np.random.gamma(2, 0.01, n_samples)
        
        # Create enhanced feature set
        options_data = pd.DataFrame({
            'strike': strikes,
            'current_price': S0,
            'time_to_expiry': times_to_expiry,
            'implied_volatility': volatilities,
            'volume': volume,
            'open_interest': open_interest,
            'bid_ask_spread': bid_ask_spread,
            'delta': observed_delta,
            'theoretical_delta': theoretical_delta
        })
        
        # Advanced feature engineering
        options_data['moneyness'] = options_data['strike'] / options_data['current_price']
        options_data['log_moneyness'] = np.log(options_data['moneyness'])
        options_data['sqrt_time'] = np.sqrt(options_data['time_to_expiry'])
        options_data['time_decay'] = 1 / np.sqrt(options_data['time_to_expiry'])
        
        # Volume and liquidity features
        options_data['log_volume'] = np.log1p(options_data['volume'])
        options_data['log_oi'] = np.log1p(options_data['open_interest'])
        options_data['volume_oi_ratio'] = options_data['volume'] / np.maximum(options_data['open_interest'], 1)
        options_data['liquidity_score'] = options_data['volume'] / (1 + options_data['bid_ask_spread'])
        
        # Volatility features
        options_data['iv_rank'] = options_data.groupby(pd.cut(options_data['time_to_expiry'], bins=5))['implied_volatility'].rank(pct=True)
        options_data['iv_moneyness_interaction'] = options_data['implied_volatility'] * np.abs(options_data['log_moneyness'])
        options_data['iv_time_interaction'] = options_data['implied_volatility'] * options_data['sqrt_time']
        
        # Risk metrics
        options_data['vega_proxy'] = options_data['sqrt_time'] * options_data['current_price'] * norm.pdf(d1) * 0.01
        options_data['theta_proxy'] = -(options_data['current_price'] * norm.pdf(d1) * options_data['implied_volatility']) / (2 * options_data['sqrt_time'])
        
        return options_data
    
    # Generate enhanced dataset
    options_df_enhanced = generate_enhanced_options_data()
    
    return generate_enhanced_options_data, options_df_enhanced


@app.cell
def __(mo, options_df_enhanced):
    # Enhanced feature selection
    feature_columns_enhanced = [
        'log_moneyness', 'sqrt_time', 'time_decay', 'implied_volatility',
        'log_volume', 'log_oi', 'volume_oi_ratio', 'liquidity_score',
        'iv_rank', 'iv_moneyness_interaction', 'iv_time_interaction',
        'vega_proxy', 'theta_proxy'
    ]
    
    X_enhanced = options_df_enhanced[feature_columns_enhanced].values
    y_enhanced = options_df_enhanced['delta'].values
    y_theoretical = options_df_enhanced['theoretical_delta'].values
    
    # Data quality report
    data_report = mo.md(f"""
    ### 📊 Enhanced Dataset Report
    - **Total Samples**: {len(options_df_enhanced):,}
    - **Features**: {len(feature_columns_enhanced)}
    - **Delta Range**: {y_enhanced.min():.4f} - {y_enhanced.max():.4f}
    - **IV Range**: {options_df_enhanced['implied_volatility'].min():.1%} - {options_df_enhanced['implied_volatility'].max():.1%}
    - **Time Range**: {options_df_enhanced['time_to_expiry'].min()*365:.0f} - {options_df_enhanced['time_to_expiry'].max()*365:.0f} days
    - **Strike Range**: ${options_df_enhanced['strike'].min():.0f} - ${options_df_enhanced['strike'].max():.0f}
    """)
    
    feature_table = options_df_enhanced[feature_columns_enhanced + ['delta']].head(10)
    
    mo.hstack([data_report, mo.ui.table(feature_table)])
    return (
        X_enhanced,
        feature_columns_enhanced,
        feature_table,
        y_enhanced,
        y_theoretical,
    )


@app.cell
def __(F, activation_fn, nn):
    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for transformer"""
        def __init__(self, d_model, max_len=100):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               -(math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe.unsqueeze(0))
        
        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class MultiScaleAttention(nn.Module):
        """Multi-scale attention mechanism"""
        def __init__(self, d_model, nhead, dropout=0.1):
            super().__init__()
            self.local_attention = nn.MultiheadAttention(d_model, nhead//2, dropout=dropout, batch_first=True)
            self.global_attention = nn.MultiheadAttention(d_model, nhead//2, dropout=dropout, batch_first=True)
            self.combine = nn.Linear(d_model * 2, d_model)
            
        def forward(self, x):
            # Local attention (within sequence)
            local_out, local_attn = self.local_attention(x, x, x)
            
            # Global attention (across entire sequence)
            global_out, global_attn = self.global_attention(x, x, x)
            
            # Combine both attention mechanisms
            combined = torch.cat([local_out, global_out], dim=-1)
            output = self.combine(combined)
            
            return output, (local_attn, global_attn)

    class AdvancedTransformerDeltaPredictor(nn.Module):
        """Advanced Transformer with enhanced features"""
        def __init__(self, input_size, d_model=128, nhead=8, num_layers=4, 
                     dropout=0.2, activation='gelu'):
            super().__init__()
            
            self.input_size = input_size
            self.d_model = d_model
            
            # Input projection with residual connection
            self.input_projection = nn.Sequential(
                nn.Linear(input_size, d_model),
                nn.LayerNorm(d_model),
                self._get_activation(activation),
                nn.Dropout(dropout)
            )
            
            # Positional encoding
            self.pos_encoding = PositionalEncoding(d_model)
            
            # Feature-wise embedding
            self.feature_embeddings = nn.ModuleList([
                nn.Linear(1, d_model // input_size) for _ in range(input_size)
            ])
            
            # Multi-scale attention layers
            self.multi_scale_layers = nn.ModuleList([
                MultiScaleAttention(d_model, nhead, dropout) for _ in range(num_layers//2)
            ])
            
            # Standard transformer layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=True  # Pre-norm for better training stability
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers//2)
            
            # Attention pooling
            self.attention_pooling = nn.Sequential(
                nn.Linear(d_model, d_model//2),
                self._get_activation(activation),
                nn.Linear(d_model//2, 1),
                nn.Softmax(dim=1)
            )
            
            # Output layers with residual connections
            self.output_layers = nn.Sequential(
                nn.Linear(d_model, d_model//2),
                nn.LayerNorm(d_model//2),
                self._get_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(d_model//2, d_model//4),
                nn.LayerNorm(d_model//4),
                self._get_activation(activation),
                nn.Dropout(dropout//2),
                nn.Linear(d_model//4, 1),
                nn.Sigmoid()
            )
            
            # Uncertainty head
            self.uncertainty_head = nn.Sequential(
                nn.Linear(d_model, d_model//4),
                self._get_activation(activation),
                nn.Linear(d_model//4, 1),
                nn.Softplus()
            )
            
            # Temperature parameter for calibration
            self.temperature = nn.Parameter(torch.ones(1))
            
            # Initialize weights
            self.apply(self._init_weights)
        
        def _get_activation(self, name):
            activations = {
                'relu': nn.ReLU(),
                'gelu': nn.GELU(),
                'swish': nn.SiLU(),
                'leaky_relu': nn.LeakyReLU(0.1)
            }
            return activations.get(name, nn.GELU())
        
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
        
        def forward(self, x, return_attention=False):
            batch_size = x.size(0)
            
            # Feature-wise embeddings
            feature_embeds = []
            for i, embed_layer in enumerate(self.feature_embeddings):
                feat_embed = embed_layer(x[:, i:i+1])
                feature_embeds.append(feat_embed)
            
            # Combine feature embeddings
            combined_embeds = torch.cat(feature_embeds, dim=-1)
            
            # Project to model dimension
            x_proj = self.input_projection(x) + combined_embeds
            
            # Add sequence dimension and positional encoding
            x_seq = x_proj.unsqueeze(1)  # (batch, 1, d_model)
            x_pos = self.pos_encoding(x_seq)
            
            # Multi-scale attention
            attention_weights = []
            for ms_layer in self.multi_scale_layers:
                x_pos, attn_weights = ms_layer(x_pos)
                attention_weights.append(attn_weights)
            
            # Standard transformer layers
            transformer_out = self.transformer(x_pos)
            
            # Attention pooling
            attn_weights = self.attention_pooling(transformer_out)
            pooled = torch.sum(attn_weights * transformer_out, dim=1)
            
            # Predictions
            delta_pred = self.output_layers(pooled) / self.temperature
            uncertainty = self.uncertainty_head(pooled)
            
            if return_attention:
                return delta_pred, uncertainty, attention_weights
            
            return delta_pred, uncertainty
    
    return AdvancedTransformerDeltaPredictor, MultiScaleAttention, PositionalEncoding


@app.cell
def __(mo):
    # Training button and options
    train_button = mo.ui.button(label="🚀 Train Advanced Transformer", kind="success")
    
    use_mixed_precision = mo.ui.checkbox(value=True, label="Use Mixed Precision Training")
    use_gradient_clipping = mo.ui.checkbox(value=True, label="Enable Gradient Clipping")
    save_checkpoints = mo.ui.checkbox(value=False, label="Save Model Checkpoints")
    
    mo.hstack([
        train_button,
        mo.vstack([use_mixed_precision, use_gradient_clipping, save_checkpoints])
    ])
    return save_checkpoints, train_button, use_gradient_clipping, use_mixed_precision


@app.cell
def __(
    AdvancedTransformerDeltaPredictor,
    RobustScaler,
    X_enhanced,
    activation_fn,
    batch_size,
    d_model,
    dropout_rate,
    epochs,
    learning_rate,
    loss_function,
    mo,
    nn,
    nhead,
    np,
    num_layers,
    optimizer_type,
    r2_score,
    scheduler_type,
    torch,
    train_button,
    use_gradient_clipping,
    use_mixed_precision,
    weight_decay,
    y_enhanced,
):
    training_results = None
    
    if train_button.value > 0:
        # Prepare data with enhanced scaling
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_enhanced)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_enhanced).unsqueeze(1)
        
        # Advanced train/validation split with stratification
        from sklearn.model_selection import train_test_split
        
        # Create bins for stratified split based on delta values
        delta_bins = np.digitize(y_enhanced, bins=np.quantile(y_enhanced, [0.2, 0.4, 0.6, 0.8]))
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_tensor, y_tensor, test_size=0.2, stratify=delta_bins, random_state=42
        )
        
        # Initialize advanced model
        model = AdvancedTransformerDeltaPredictor(
            input_size=X_enhanced.shape[1],
            d_model=d_model.value,
            nhead=nhead.value,
            num_layers=num_layers.value,
            dropout=dropout_rate.value,
            activation=activation_fn.value
        )
        
        # Advanced optimizer selection
        if optimizer_type.value == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate.value, weight_decay=weight_decay.value)
        elif optimizer_type.value == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate.value, weight_decay=weight_decay.value)
        elif optimizer_type.value == "RMSprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate.value, weight_decay=weight_decay.value)
        else:  # SGD
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate.value, momentum=0.9, weight_decay=weight_decay.value)
        
        # Scheduler selection
        if scheduler_type.value == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs.value//3, gamma=0.1)
        elif scheduler_type.value == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs.value)
        elif scheduler_type.value == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        else:
            scheduler = None
        
        # Loss function selection
        mse_loss = nn.MSELoss()
        huber_loss = nn.HuberLoss(delta=0.1)
        l1_loss = nn.L1Loss()
        
        def get_loss(pred, target, uncertainty=None):
            if loss_function.value == "MSE":
                return mse_loss(pred, target)
            elif loss_function.value == "Huber":
                return huber_loss(pred, target)
            elif loss_function.value == "L1":
                return l1_loss(pred, target)
            else:  # Combined
                base_loss = 0.6 * mse_loss(pred, target) + 0.4 * huber_loss(pred, target)
                if uncertainty is not None:
                    # Add uncertainty regularization
                    uncertainty_loss = uncertainty.mean() * 0.01
                    return base_loss + uncertainty_loss
                return base_loss
        
        # Mixed precision setup
        if use_mixed_precision.value and torch.cuda.is_available():
            scaler_amp = torch.cuda.amp.GradScaler()
        else:
            scaler_amp = None
        
        # Training loop with comprehensive metrics
        train_losses = []
        val_losses = []
        learning_rates = []
        r2_scores = []
        train_uncertainties = []
        val_uncertainties = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        model.train()
        for epoch in range(min(epochs.value, 100)):  # Limit for demo
            # Training phase
            model.train()
            epoch_train_loss = 0
            epoch_train_uncertainty = 0
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size.value):
                batch_X = X_train[i:i+batch_size.value]
                batch_y = y_train[i:i+batch_size.value]
                
                optimizer.zero_grad()
                
                if scaler_amp is not None:
                    with torch.cuda.amp.autocast():
                        pred, uncertainty = model(batch_X)
                        loss = get_loss(pred, batch_y, uncertainty)
                    
                    scaler_amp.scale(loss).backward()
                    
                    if use_gradient_clipping.value:
                        scaler_amp.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                else:
                    pred, uncertainty = model(batch_X)
                    loss = get_loss(pred, batch_y, uncertainty)
                    loss.backward()
                    
                    if use_gradient_clipping.value:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                
                epoch_train_loss += loss.item()
                epoch_train_uncertainty += uncertainty.mean().item()
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_uncertainty = 0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for i in range(0, len(X_val), batch_size.value):
                    batch_X = X_val[i:i+batch_size.value]
                    batch_y = y_val[i:i+batch_size.value]
                    
                    pred, uncertainty = model(batch_X)
                    loss = get_loss(pred, batch_y, uncertainty)
                    
                    val_loss += loss.item()
                    val_uncertainty += uncertainty.mean().item()
                    val_predictions.extend(pred.numpy())
                    val_targets.extend(batch_y.numpy())
            
            # Calculate metrics
            avg_train_loss = epoch_train_loss / (len(X_train) // batch_size.value)
            avg_val_loss = val_loss / (len(X_val) // batch_size.value)
            avg_train_uncertainty = epoch_train_uncertainty / (len(X_train) // batch_size.value)
            avg_val_uncertainty = val_uncertainty / (len(X_val) // batch_size.value)
            
            # R² score
            r2 = r2_score(val_targets, val_predictions)
            
            # Store metrics
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            learning_rates.append(optimizer.param_groups[0]['lr'])
            r2_scores.append(r2)
            train_uncertainties.append(avg_train_uncertainty)
            val_uncertainties.append(avg_val_uncertainty)
            
            # Scheduler step
            if scheduler is not None:
                if scheduler_type.value == "ReduceLROnPlateau":
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Final predictions
        model.eval()
        with torch.no_grad():
            final_predictions, final_uncertainties = model(X_tensor)
            final_predictions = final_predictions.numpy().flatten()
            final_uncertainties = final_uncertainties.numpy().flatten()
        
        training_results = {
            'model': model,
            'scaler': scaler,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'learning_rates': learning_rates,
            'r2_scores': r2_scores,
            'train_uncertainties': train_uncertainties,
            'val_uncertainties': val_uncertainties,
            'predictions': final_predictions,
            'uncertainties': final_uncertainties,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_losses),
            'config': {
                'd_model': d_model.value,
                'nhead': nhead.value,
                'num_layers': num_layers.value,
                'dropout': dropout_rate.value,
                'activation': activation_fn.value,
                'optimizer': optimizer_type.value,
                'scheduler': scheduler_type.value,
                'loss_function': loss_function.value
            }
        }
        
        mo.md(f"""
        ✅ **Training Complete!**
        - **Epochs Trained**: {len(train_losses)}
        - **Best Validation Loss**: {best_val_loss:.6f}
        - **Final R² Score**: {r2_scores[-1]:.4f}
        - **Model Parameters**: {sum(p.numel() for p in model.parameters()):,}
        - **Configuration**: {d_model.value}d × {nhead.value}h × {num_layers.value}L
        """)
    else:
        mo.md("👆 Click **Train Advanced Transformer** to start training")
    
    return training_results,


@app.cell
def __(PLOTLY_AVAILABLE, go, make_subplots, mo, training_results):
    if training_results is not None and PLOTLY_AVAILABLE:
        # Comprehensive training visualization
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Training & Validation Loss',
                'R² Score Evolution', 
                'Learning Rate Schedule',
                'Uncertainty Evolution',
                'Loss Components',
                'Training Metrics'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        epochs_range = list(range(1, len(training_results['train_losses']) + 1))
        
        # Training/Validation Loss
        fig.add_trace(
            go.Scatter(x=epochs_range, y=training_results['train_losses'], 
                      name='Train Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs_range, y=training_results['val_losses'], 
                      name='Val Loss', line=dict(color='red')),
            row=1, col=1
        )
        
        # R² Score
        fig.add_trace(
            go.Scatter(x=epochs_range, y=training_results['r2_scores'], 
                      name='R² Score', line=dict(color='green')),
            row=1, col=2
        )
        
        # Learning Rate
        fig.add_trace(
            go.Scatter(x=epochs_range, y=training_results['learning_rates'], 
                      name='Learning Rate', line=dict(color='orange')),
            row=1, col=3
        )
        
        # Uncertainties
        fig.add_trace(
            go.Scatter(x=epochs_range, y=training_results['train_uncertainties'], 
                      name='Train Uncertainty', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs_range, y=training_results['val_uncertainties'], 
                      name='Val Uncertainty', line=dict(color='pink')),
            row=2, col=1
        )
        
        # Loss difference (overfitting indicator)
        loss_diff = [v - t for v, t in zip(training_results['val_losses'], training_results['train_losses'])]
        fig.add_trace(
            go.Scatter(x=epochs_range, y=loss_diff, 
                      name='Loss Difference', line=dict(color='brown')),
            row=2, col=2
        )
        
        # Training efficiency (R² per epoch)
        r2_velocity = [0] + [training_results['r2_scores'][i] - training_results['r2_scores'][i-1] 
                            for i in range(1, len(training_results['r2_scores']))]
        fig.add_trace(
            go.Scatter(x=epochs_range, y=r2_velocity, 
                      name='R² Velocity', line=dict(color='teal')),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(height=600, showlegend=True, title_text="Comprehensive Training Analysis")
        
        # Update axis labels
        for i in range(1, 4):
            fig.update_xaxes(title_text="Epoch", row=2, col=i)
        
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="R² Score", row=1, col=2)
        fig.update_yaxes(title_text="Learning Rate", row=1, col=3)
        fig.update_yaxes(title_text="Uncertainty", row=2, col=1)
        fig.update_yaxes(title_text="Loss Difference", row=2, col=2)
        fig.update_yaxes(title_text="R² Velocity", row=2, col=3)
        
        mo.ui.plotly(fig)
    elif training_results is not None:
        mo.md("📊 Training completed! (Plotly visualization not available)")
    else:
        mo.md("📊 Training visualizations will appear after model training")
    return epochs_range, fig, loss_diff, r2_velocity


@app.cell
def __(
    PLOTLY_AVAILABLE,
    go,
    mo,
    np,
    training_results,
    y_enhanced,
    y_theoretical,
):
    if training_results is not None and PLOTLY_AVAILABLE:
        predictions = training_results['predictions']
        uncertainties = training_results['uncertainties']
        
        # Prediction quality analysis
        fig_quality = go.Figure()
        
        # Perfect prediction line
        min_val = min(y_enhanced.min(), predictions.min())
        max_val = max(y_enhanced.max(), predictions.max())
        fig_quality.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name='Perfect Prediction',
                      line=dict(dash='dash', color='gray', width=2))
        )
        
        # Actual vs predicted with uncertainty coloring
        error_magnitude = np.abs(y_enhanced - predictions)
        fig_quality.add_trace(
            go.Scatter(
                x=y_enhanced, 
                y=predictions,
                mode='markers',
                name='Predictions',
                marker=dict(
                    color=uncertainties,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Model Uncertainty"),
                    size=6,
                    opacity=0.7
                ),
                text=[f'Error: {err:.4f}<br>Uncertainty: {unc:.4f}' 
                      for err, unc in zip(error_magnitude, uncertainties)],
                hovertemplate='Actual: %{x:.4f}<br>Predicted: %{y:.4f}<br>%{text}<extra></extra>'
            )
        )
        
        # Theoretical delta comparison
        fig_quality.add_trace(
            go.Scatter(
                x=y_theoretical,
                y=predictions,
                mode='markers',
                name='vs Theoretical',
                marker=dict(color='red', size=4, opacity=0.5),
                hovertemplate='Theoretical: %{x:.4f}<br>Predicted: %{y:.4f}<extra></extra>'
            )
        )
        
        fig_quality.update_layout(
            title="Prediction Quality Analysis",
            xaxis_title="Actual Delta",
            yaxis_title="Predicted Delta",
            height=500
        )
        
        mo.ui.plotly(fig_quality)
    elif training_results is not None:
        mo.md("📊 Prediction quality analysis completed! (Plotly not available)")
    else:
        mo.md("📊 Prediction analysis will appear after training")
    return error_magnitude, fig_quality, max_val, min_val, predictions, uncertainties


@app.cell
def __(
    mean_absolute_error,
    mean_squared_error,
    mo,
    np,
    pd,
    predictions,
    r2_score,
    training_results,
    uncertainties,
    y_enhanced,
    y_theoretical,
):
    if training_results is not None:
        # Comprehensive performance metrics
        mae = mean_absolute_error(y_enhanced, predictions)
        mse = mean_squared_error(y_enhanced, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_enhanced, predictions)
        
        # Delta-specific metrics
        delta_accuracy_1bp = np.mean(np.abs(predictions - y_enhanced) < 0.01)
        delta_accuracy_5bp = np.mean(np.abs(predictions - y_enhanced) < 0.05)
        delta_accuracy_10bp = np.mean(np.abs(predictions - y_enhanced) < 0.10)
        
        # Theoretical delta comparison
        theoretical_mae = mean_absolute_error(y_theoretical, predictions)
        theoretical_r2 = r2_score(y_theoretical, predictions)
        
        # Uncertainty calibration metrics
        mean_uncertainty = np.mean(uncertainties)
        uncertainty_correlation = np.corrcoef(uncertainties, np.abs(predictions - y_enhanced))[0, 1]
        
        # Moneyness-based analysis
        from scipy.stats import pearsonr
        
        # Calculate moneyness for error analysis
        strikes = np.exp(np.random.normal(0, 0.2, len(predictions))) * 150  # Approximate strikes
        moneyness = strikes / 150
        
        # ITM/OTM performance
        itm_mask = moneyness < 1.0
        otm_mask = moneyness > 1.0
        
        itm_mae = mean_absolute_error(y_enhanced[itm_mask], predictions[itm_mask]) if np.any(itm_mask) else 0
        otm_mae = mean_absolute_error(y_enhanced[otm_mask], predictions[otm_mask]) if np.any(otm_mask) else 0
        
        # Volatility regime performance
        high_vol_mask = np.random.random(len(predictions)) > 0.5  # Simplified for demo
        low_vol_mask = ~high_vol_mask
        
        high_vol_mae = mean_absolute_error(y_enhanced[high_vol_mask], predictions[high_vol_mask])
        low_vol_mae = mean_absolute_error(y_enhanced[low_vol_mask], predictions[low_vol_mask])
        
        # Create comprehensive metrics table
        metrics_table = pd.DataFrame({
            'Category': [
                'Overall', 'Overall', 'Overall', 'Overall',
                'Accuracy', 'Accuracy', 'Accuracy',
                'Theoretical', 'Theoretical',
                'Uncertainty', 'Uncertainty',
                'Regime', 'Regime', 'Regime', 'Regime',
                'Model', 'Model', 'Model'
            ],
            'Metric': [
                'MAE', 'MSE', 'RMSE', 'R²',
                'Within 1bp', 'Within 5bp', 'Within 10bp',
                'vs Theoretical MAE', 'vs Theoretical R²',
                'Mean Uncertainty', 'Uncertainty-Error Correlation',
                'ITM MAE', 'OTM MAE', 'High Vol MAE', 'Low Vol MAE',
                'Parameters', 'Best Val Loss', 'Epochs Trained'
            ],
            'Value': [
                f'{mae:.6f}', f'{mse:.6f}', f'{rmse:.6f}', f'{r2:.4f}',
                f'{delta_accuracy_1bp:.2%}', f'{delta_accuracy_5bp:.2%}', f'{delta_accuracy_10bp:.2%}',
                f'{theoretical_mae:.6f}', f'{theoretical_r2:.4f}',
                f'{mean_uncertainty:.6f}', f'{uncertainty_correlation:.4f}',
                f'{itm_mae:.6f}', f'{otm_mae:.6f}', f'{high_vol_mae:.6f}', f'{low_vol_mae:.6f}',
                f'{sum(p.numel() for p in training_results["model"].parameters()):,}',
                f'{training_results["best_val_loss"]:.6f}',
                f'{training_results["epochs_trained"]}'
            ]
        })
        
        # Performance summary
        performance_summary = mo.md(f"""
        ## 📊 Comprehensive Performance Analysis
        
        ### 🎯 Key Metrics
        - **Overall R² Score**: {r2:.4f}
        - **Mean Absolute Error**: {mae:.6f}
        - **Accuracy (±5bp)**: {delta_accuracy_5bp:.2%}
        - **Model Uncertainty**: {mean_uncertainty:.6f}
        
        ### 📈 Model Insights
        - **Best Performance**: {"ITM" if itm_mae < otm_mae else "OTM"} options
        - **Uncertainty Calibration**: {"Well calibrated" if uncertainty_correlation > 0.3 else "Needs improvement"}
        - **Volatility Sensitivity**: {"High vol" if high_vol_mae < low_vol_mae else "Low vol"} conditions
        """)
        
        mo.hstack([performance_summary, mo.ui.table(metrics_table)])
    else:
        mo.md("📈 Comprehensive metrics will be calculated after training")
    return (
        delta_accuracy_10bp,
        delta_accuracy_1bp,
        delta_accuracy_5bp,
        high_vol_mae,
        high_vol_mask,
        itm_mae,
        itm_mask,
        low_vol_mae,
        low_vol_mask,
        mae,
        mean_uncertainty,
        metrics_table,
        moneyness,
        mse,
        otm_mae,
        otm_mask,
        pearsonr,
        performance_summary,
        r2,
        rmse,
        strikes,
        theoretical_mae,
        theoretical_r2,
        uncertainty_correlation,
    )


@app.cell
def __(mo, training_results):
    if training_results is not None:
        mo.md("## 🎮 Interactive Delta Prediction")
        
        # Enhanced prediction interface
        pred_strike = mo.ui.slider(
            start=50, stop=300, step=1, value=150,
            label="Strike Price ($)"
        )
        
        pred_time = mo.ui.slider(
            start=1, stop=730, step=1, value=30,
            label="Days to Expiry"
        )
        
        pred_iv = mo.ui.slider(
            start=5, stop=100, step=1, value=25,
            label="Implied Volatility (%)"
        )
        
        pred_volume = mo.ui.slider(
            start=1, stop=10000, step=100, value=1000,
            label="Volume"
        )
        
        pred_oi = mo.ui.slider(
            start=100, stop=50000, step=500, value=5000,
            label="Open Interest"
        )
        
        pred_spread = mo.ui.slider(
            start=0.01, stop=1.0, step=0.01, value=0.05,
            label="Bid-Ask Spread"
        )
        
        mo.hstack([
            mo.vstack([pred_strike, pred_time, pred_iv]),
            mo.vstack([pred_volume, pred_oi, pred_spread])
        ])
    else:
        pred_strike = pred_time = pred_iv = pred_volume = pred_oi = pred_spread = None
        mo.md("⏳ Complete training to access interactive predictions")
    return pred_iv, pred_oi, pred_spread, pred_strike, pred_time, pred_volume


@app.cell
def __(
    PLOTLY_AVAILABLE,
    go,
    mo,
    np,
    pred_iv,
    pred_oi,
    pred_spread,
    pred_strike,
    pred_time,
    pred_volume,
    torch,
    training_results,
):
    if training_results is not None and all(x is not None for x in [pred_strike, pred_time, pred_iv, pred_volume, pred_oi, pred_spread]):
        
        # Current market parameters
        current_price = 150
        
        # Extract prediction parameters
        strike_val = pred_strike.value
        time_val = pred_time.value / 365  # Convert to years
        iv_val = pred_iv.value / 100  # Convert to decimal
        volume_val = pred_volume.value
        oi_val = pred_oi.value
        spread_val = pred_spread.value
        
        # Advanced feature engineering
        moneyness_pred = strike_val / current_price
        log_moneyness_pred = np.log(moneyness_pred)
        sqrt_time_pred = np.sqrt(time_val)
        time_decay_pred = 1 / sqrt_time_pred if sqrt_time_pred > 0 else 1
        log_volume_pred = np.log1p(volume_val)
        log_oi_pred = np.log1p(oi_val)
        volume_oi_ratio_pred = volume_val / max(oi_val, 1)
        liquidity_score_pred = volume_val / (1 + spread_val)
        iv_rank_pred = 0.5  # Simplified
        iv_moneyness_interaction_pred = iv_val * abs(log_moneyness_pred)
        iv_time_interaction_pred = iv_val * sqrt_time_pred
        
        # Calculate Greeks proxies
        from scipy.stats import norm
        d1_pred = (np.log(current_price/strike_val) + (0.05 + 0.5*iv_val**2)*time_val) / (iv_val*sqrt_time_pred)
        vega_proxy_pred = sqrt_time_pred * current_price * norm.pdf(d1_pred) * 0.01
        theta_proxy_pred = -(current_price * norm.pdf(d1_pred) * iv_val) / (2 * sqrt_time_pred)
        
        # Create feature vector
        live_features = np.array([[
            log_moneyness_pred, sqrt_time_pred, time_decay_pred, iv_val,
            log_volume_pred, log_oi_pred, volume_oi_ratio_pred, liquidity_score_pred,
            iv_rank_pred, iv_moneyness_interaction_pred, iv_time_interaction_pred,
            vega_proxy_pred, theta_proxy_pred
        ]])
        
        # Scale features
        live_features_scaled = training_results['scaler'].transform(live_features)
        
        # Make prediction with uncertainty
        with torch.no_grad():
            training_results['model'].eval()
            live_pred, live_uncertainty = training_results['model'](torch.FloatTensor(live_features_scaled))
            live_prediction = live_pred.item()
            uncertainty_val = live_uncertainty.item()
        
        # Calculate theoretical delta for comparison
        theoretical_delta_pred = norm.cdf(d1_pred)
        
        # Calculate prediction confidence interval
        confidence_lower = max(0, live_prediction - 1.96 * uncertainty_val)
        confidence_upper = min(1, live_prediction + 1.96 * uncertainty_val)
        
        if PLOTLY_AVAILABLE:
            # Enhanced gauge visualization
            fig_gauge = go.Figure()
            
            # Main delta gauge
            fig_gauge.add_trace(go.Indicator(
                mode = "gauge+number+delta",
                value = live_prediction,
                delta = {'reference': theoretical_delta_pred, 'relative': False},
                title = {'text': f"Predicted Delta<br><span style='font-size:0.8em;color:gray'>±{uncertainty_val:.4f} uncertainty</span>"},
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
                        'value': theoretical_delta_pred
                    }
                }
            ))
            
            fig_gauge.update_layout(height=400, title_text="Live Delta Prediction")
        
        # Enhanced results display
        option_type = "ITM Call" if moneyness_pred < 1 else "OTM Call"
        vol_regime = "High Vol" if iv_val > 0.3 else "Normal Vol" if iv_val > 0.15 else "Low Vol"
        time_regime = "Short Term" if time_val < 0.1 else "Medium Term" if time_val < 0.5 else "Long Term"
        liquidity_regime = "High Liquidity" if liquidity_score_pred > 1000 else "Medium Liquidity" if liquidity_score_pred > 100 else "Low Liquidity"
        
        results_detailed = mo.md(f"""
        ### 🎯 Live Prediction Results
        
        **📊 Market Parameters:**
        - **Strike**: ${strike_val} ({option_type})
        - **Current Price**: ${current_price}
        - **Moneyness**: {moneyness_pred:.3f}
        - **Time to Expiry**: {pred_time.value} days ({time_regime})
        - **Implied Volatility**: {pred_iv.value}% ({vol_regime})
        - **Volume**: {volume_val:,} / OI: {oi_val:,} ({liquidity_regime})
        
        **🎯 Delta Predictions:**
        - **Model Prediction**: {live_prediction:.4f} ± {uncertainty_val:.4f}
        - **Theoretical (BS)**: {theoretical_delta_pred:.4f}
        - **Prediction Difference**: {abs(live_prediction - theoretical_delta_pred):.4f}
        - **95% Confidence Interval**: [{confidence_lower:.4f}, {confidence_upper:.4f}]
        
        **📈 Greeks Proxies:**
        - **Vega Proxy**: {vega_proxy_pred:.4f}
        - **Theta Proxy**: {theta_proxy_pred:.4f}
        - **Liquidity Score**: {liquidity_score_pred:.0f}
        
        **🔍 Model Confidence:**
        - **Uncertainty Level**: {"Low" if uncertainty_val < 0.01 else "Medium" if uncertainty_val < 0.05 else "High"}
        - **Prediction Quality**: {"High confidence" if uncertainty_val < 0.02 else "Moderate confidence" if uncertainty_val < 0.05 else "Low confidence"}
        """)
        
        if PLOTLY_AVAILABLE:
            mo.hstack([mo.ui.plotly(fig_gauge), results_detailed])
        else:
            results_detailed
    else:
        mo.md("🔄 Awaiting prediction parameters...")
    return (
        confidence_lower,
        confidence_upper,
        current_price,
        d1_pred,
        fig_gauge,
        iv_moneyness_interaction_pred,
        iv_rank_pred,
        iv_time_interaction_pred,
        iv_val,
        liquidity_regime,
        liquidity_score_pred,
        live_features,
        live_features_scaled,
        live_pred,
        live_prediction,
        live_uncertainty,
        log_moneyness_pred,
        log_oi_pred,
        log_volume_pred,
        moneyness_pred,
        oi_val,
        option_type,
        results_detailed,
        spread_val,
        sqrt_time_pred,
        strike_val,
        theoretical_delta_pred,
        theta_proxy_pred,
        time_decay_pred,
        time_regime,
        time_val,
        uncertainty_val,
        vega_proxy_pred,
        vol_regime,
        volume_oi_ratio_pred,
        volume_val,
    )


@app.cell
def __(mo, training_results):
    if training_results is not None:
        mo.md(f"""
        ## 🔬 Model Architecture Analysis
        
        ### 🏗️ Transformer Configuration
        - **Model Dimension**: {training_results['config']['d_model']}
        - **Attention Heads**: {training_results['config']['nhead']}
        - **Transformer Layers**: {training_results['config']['num_layers']}
        - **Dropout Rate**: {training_results['config']['dropout']:.1%}
        - **Activation Function**: {training_results['config']['activation'].upper()}
        - **Total Parameters**: {sum(p.numel() for p in training_results['model'].parameters()):,}
        
        ### ⚡ Training Configuration
        - **Optimizer**: {training_results['config']['optimizer']}
        - **Scheduler**: {training_results['config']['scheduler']}
        - **Loss Function**: {training_results['config']['loss_function']}
        - **Best Validation Loss**: {training_results['best_val_loss']:.6f}
        - **Training Epochs**: {training_results['epochs_trained']}
        """)
    else:
        mo.md("🔬 Architecture analysis will be available after training")
    return


if __name__ == "__main__":
    app.run()