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