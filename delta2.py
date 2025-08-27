# most pytorch elements
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv
import warnings
import json
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

class EnhancedDeltaDataset(Dataset):
    """Enhanced dataset with optional data augmentation"""
    def __init__(self, features, targets, augment=False):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.augment = augment
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        targets = self.targets[idx]
        
        if self.augment and torch.rand(1) > 0.5:
            # Add small gaussian noise for regularization
            noise = torch.normal(0, 0.01, features.shape)
            features = features + noise
            
        return features, targets

class TransformerDeltaPredictor(nn.Module):
    """Transformer-based delta predictor with attention mechanisms"""
    def __init__(self, input_size=7, d_model=128, nhead=8, num_layers=4, dropout=0.2):
        super(TransformerDeltaPredictor, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(100, d_model))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Feature-wise attention
        self.feature_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(), 
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Project input features to transformer dimension
        x = self.input_projection(x)  # (batch, d_model)
        
        # Add positional encoding and expand for sequence modeling
        x = x.unsqueeze(1) + self.pos_encoding[0:1, :]  # (batch, 1, d_model)
        
        # Self-attention through transformer
        transformer_out = self.transformer(x)  # (batch, 1, d_model)
        
        # Feature attention mechanism
        attended_out, attention_weights = self.feature_attention(
            transformer_out, transformer_out, transformer_out
        )
        
        # Combine transformer and attention outputs
        combined = transformer_out + attended_out  # Residual connection
        
        # Pool and output
        pooled = combined.squeeze(1)  # (batch, d_model)
        
        # Temperature-scaled output
        output = self.output_layers(pooled)
        return output / self.temperature

class CNNFeatureExtractor(nn.Module):
    """1D CNN for extracting local patterns in option features"""
    def __init__(self, input_size=7, hidden_channels=64, output_size=32):
        super(CNNFeatureExtractor, self).__init__()
        
        # 1D Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(output_size)
        )
        
        self.output_projection = nn.Linear(hidden_channels * 2 * output_size, output_size)
        
    def forward(self, x):
        # Reshape for 1D conv: (batch, 1, features)
        x = x.unsqueeze(1)
        
        # Extract features through conv layers
        conv_out = self.conv_layers(x)  # (batch, hidden_channels*2, output_size)
        
        # Flatten and project
        flattened = conv_out.view(conv_out.size(0), -1)
        return self.output_projection(flattened)

class LSTMDeltaPredictor(nn.Module):
    """LSTM-based model for sequential option data modeling"""
    def __init__(self, input_size=7, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMDeltaPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output layers
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Reshape for LSTM: (batch, seq_len=1, features)
        x = x.unsqueeze(1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        
        # Attention weights
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        
        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size*2)
        
        return self.output(context)

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        
        # Residual connection
        out += residual
        return F.relu(out)

class EnsembleDeltaPredictor(nn.Module):
    """Ensemble model combining multiple architectures"""
    def __init__(self, input_size=7, hidden_size=128):
        super(EnsembleDeltaPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Multiple model architectures
        self.transformer_model = TransformerDeltaPredictor(input_size, d_model=hidden_size)
        self.lstm_model = LSTMDeltaPredictor(input_size, hidden_size=hidden_size)
        self.cnn_extractor = CNNFeatureExtractor(input_size, output_size=hidden_size//2)
        
        # Traditional feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            ResidualBlock(hidden_size, 0.1),
            ResidualBlock(hidden_size, 0.1),
            nn.Linear(hidden_size, hidden_size//2),
            nn.GELU(),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()
        )
        
        # Feature fusion network
        fusion_input_size = hidden_size//2 + 1  # CNN features + feedforward output
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()
        )
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Confidence predictors
        self.confidence_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(input_size, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
            for _ in range(3)
        ])
    
    def forward(self, x):
        # Get predictions from each model
        transformer_pred = self.transformer_model(x)
        lstm_pred = self.lstm_model(x)
        
        # CNN feature extraction and feedforward prediction
        cnn_features = self.cnn_extractor(x)
        ff_pred = self.feedforward(x)
        
        # Feature fusion
        fused_features = torch.cat([cnn_features, ff_pred], dim=1)
        fusion_pred = self.feature_fusion(fused_features)
        
        # Confidence-weighted ensemble
        confidences = torch.stack([
            self.confidence_nets[0](x),
            self.confidence_nets[1](x),
            self.confidence_nets[2](x)
        ], dim=2).squeeze(1)  # (batch, 3)
        
        # Normalize ensemble weights
        normalized_weights = F.softmax(self.ensemble_weights, dim=0)
        
        # Weighted predictions
        predictions = torch.stack([transformer_pred, lstm_pred, fusion_pred], dim=2).squeeze(1)
        
        # Confidence-adjusted weights
        adjusted_weights = normalized_weights.unsqueeze(0) * confidences
        adjusted_weights = adjusted_weights / adjusted_weights.sum(dim=1, keepdim=True)
        
        # Final ensemble prediction
        ensemble_pred = torch.sum(predictions * adjusted_weights, dim=1, keepdim=True)
        
        return ensemble_pred

class EnhancedLiveDeltaPredictor:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.scaler = RobustScaler()
        self.model = None
        self.uncertainty_model = None
        self.model_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_stock_price = None
        self.training_history = {'loss': [], 'val_loss': [], 'lr': []}
        
        print(f"PyTorch Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
    
    def get_sample_data(self):
        """Get real options data from Alpha Vantage API"""
        print("Retrieving real options data from Alpha Vantage...")
        
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required. Set ALPHA_VANTAGE_API_KEY environment variable.")
        
        try:
            # Try to get real options data
            params = {
                'function': 'HISTORICAL_OPTIONS',
                'symbol': 'AAPL',
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                return self.parse_alpha_vantage_data(data['data'])
            else:
                # If no options data, try stock data instead
                print("No options data available, attempting to use stock data for demo...")
                return self.get_stock_based_data()
                
        except Exception as e:
            print(f"API request failed: {str(e)}")
            raise RuntimeError(f"Could not retrieve real market data: {str(e)}")
    
    def parse_alpha_vantage_data(self, raw_data):
        """Parse Alpha Vantage options data"""
        options_list = []
        
        for item in raw_data:
            try:
                options_list.append({
                    'symbol': item.get('symbol'),
                    'type': item.get('type', 'call'),
                    'strike': float(item.get('strike', 0)),
                    'expiration': item.get('expiration'),
                    'bid': float(item.get('bid', 0)),
                    'ask': float(item.get('ask', 0)),
                    'volume': int(item.get('volume', 0)),
                    'open_interest': int(item.get('open_interest', 0)),
                    'implied_volatility': float(item.get('implied_volatility', 0)),
                    'delta': float(item.get('delta', 0))
                })
            except (ValueError, TypeError) as e:
                continue  # Skip malformed records
        
        if len(options_list) < 50:
            raise ValueError(f"Insufficient options data: only {len(options_list)} records")
        
        return pd.DataFrame(options_list)
    
    def get_stock_based_data(self):
        """Fallback: Get stock data and create basic options framework for demo"""
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': 'AAPL',
            'apikey': self.api_key,
            'outputsize': 'compact'
        }
        
        response = requests.get(self.base_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if 'Time Series (Daily)' not in data:
            raise ValueError("No stock data available")
        
        # Get recent stock price
        time_series = data['Time Series (Daily)']
        latest_date = max(time_series.keys())
        current_price = float(time_series[latest_date]['4. close'])
        
        print(f"Note: Using stock price ${current_price:.2f} for demo framework")
        print("Warning: This creates a framework only - real options data needed for production")
        
        # Create minimal framework - NOT for actual trading
        demo_options = []
        strikes = np.arange(current_price * 0.85, current_price * 1.15, 5)
        expirations = [7, 14, 30, 45, 60]
        
        for strike in strikes:
            for days in expirations:
                # Basic Black-Scholes approximation for demo structure only
                moneyness = strike / current_price
                time_factor = np.sqrt(days / 365)
                
                # Rough delta approximation (NOT for real trading)
                if moneyness <= 1:
                    approx_delta = 0.5 + 0.3 * (1 - moneyness) + 0.1 * time_factor
                else:
                    approx_delta = 0.5 - 0.3 * (moneyness - 1) + 0.1 * time_factor
                
                approx_delta = max(0.01, min(0.99, approx_delta))
                
                demo_options.append({
                    'symbol': 'AAPL',
                    'type': 'call',
                    'strike': strike,
                    'expiration': (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d'),
                    'bid': max(0.01, current_price - strike + np.random.uniform(-2, 2)),
                    'ask': max(0.05, current_price - strike + np.random.uniform(-1, 3)),
                    'volume': np.random.randint(1, 1000),
                    'open_interest': np.random.randint(10, 5000),
                    'implied_volatility': np.random.uniform(0.15, 0.45),
                    'delta': approx_delta
                })
        
        return pd.DataFrame(demo_options)
    
    def prepare_enhanced_training_data(self, options_df):
        """Enhanced feature engineering with real market data only"""
        print("Preparing enhanced training data...")
        
        # Filter for valid call options only
        calls = options_df[
            (options_df['type'] == 'call') & 
            (options_df['delta'] > 0.01) & 
            (options_df['delta'] < 0.99) &
            (options_df['implied_volatility'] > 0.05) &
            (options_df['implied_volatility'] < 1.0) &
            (options_df['volume'] > 0) &
            (options_df['strike'] > 0)
        ].copy()
        
        if len(calls) < 100:
            raise ValueError(f"Insufficient clean data: only {len(calls)} valid call options")
        
        # Parse expiration dates
        try:
            calls['expiration_date'] = pd.to_datetime(calls['expiration'])
            today = datetime.now()
            calls['days_to_expiry'] = (calls['expiration_date'] - today).dt.days
            
            # Remove expired or invalid expiry dates
            calls = calls[calls['days_to_expiry'] > 0]
            calls = calls[calls['days_to_expiry'] <= 365]
            
        except Exception as e:
            raise ValueError(f"Failed to process expiration dates: {e}")
        
        if len(calls) < 100:
            raise ValueError("Insufficient data after date filtering")
        
        # Estimate current stock price
        atm_options = calls[calls['delta'].between(0.45, 0.55)]
        if len(atm_options) > 0:
            self.current_stock_price = atm_options['strike'].median()
        else:
            self.current_stock_price = (calls['strike'] * calls['delta']).sum() / calls['delta'].sum()
        
        print(f"Estimated stock price: ${self.current_stock_price:.2f}")
        
        # Enhanced feature engineering
        calls['moneyness'] = calls['strike'] / self.current_stock_price
        calls['log_moneyness'] = np.log(calls['moneyness'])
        calls['sqrt_time'] = np.sqrt(calls['days_to_expiry'] / 365)
        
        # Market microstructure features
        calls['bid_ask_spread'] = calls['ask'] - calls['bid']
        calls['mid_price'] = (calls['bid'] + calls['ask']) / 2
        calls['volume_oi_ratio'] = calls['volume'] / np.maximum(calls['open_interest'], 1)
        calls['log_volume'] = np.log1p(calls['volume'])
        calls['log_open_interest'] = np.log1p(calls['open_interest'])
        
        # IV ranking
        calls['iv_rank'] = calls.groupby('expiration')['implied_volatility'].rank(pct=True)
        
        # Time-scaled features
        calls['moneyness_time'] = calls['moneyness'] * calls['sqrt_time'] 
        calls['iv_time'] = calls['implied_volatility'] * calls['sqrt_time']
        calls['volume_time'] = calls['log_volume'] * calls['sqrt_time']
        
        # Define feature columns
        feature_columns = [
            'log_moneyness',
            'sqrt_time', 
            'implied_volatility',
            'log_volume',
            'volume_oi_ratio',
            'iv_rank',
            'moneyness_time'
        ]
        
        # Clean data
        calls = calls.dropna(subset=feature_columns + ['delta'])
        calls = calls[np.isfinite(calls[feature_columns + ['delta']]).all(axis=1)]
        
        # Remove outliers using IQR method
        for col in feature_columns:
            Q1 = calls[col].quantile(0.25)
            Q3 = calls[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            calls = calls[(calls[col] >= lower_bound) & (calls[col] <= upper_bound)]
        
        if len(calls) < 100:
            raise ValueError("Insufficient clean data after outlier removal")
        
        features = calls[feature_columns].values
        targets = calls['delta'].values.reshape(-1, 1)
        
        print(f"Training data prepared: {len(features)} samples, {features.shape[1]} features")
        print(f"Delta range: {targets.min():.4f} - {targets.max():.4f}")
        print(f"IV range: {calls['implied_volatility'].min():.2%} - {calls['implied_volatility'].max():.2%}")
        
        return features, targets
    
    def train_advanced_model(self, features, targets, epochs=100, batch_size=64, patience=15):
        """Advanced training with ensemble model"""
        print(f"Training ensemble model ({epochs} epochs, batch_size={batch_size})...")
        
        # Data scaling
        features_scaled = self.scaler.fit_transform(features)
        
        # Convert to tensors
        features_tensor = torch.from_numpy(features_scaled.astype(np.float32))
        targets_tensor = torch.from_numpy(targets.astype(np.float32))
        
        # Split data
        indices = torch.randperm(len(features_tensor))
        split_idx = int(0.8 * len(features_tensor))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        X_train = features_tensor[train_indices]
        y_train = targets_tensor[train_indices]
        X_val = features_tensor[val_indices]
        y_val = targets_tensor[val_indices]
        
        # Create datasets
        train_dataset = EnhancedDeltaDataset(X_train.numpy(), y_train.numpy(), augment=True)
        val_dataset = EnhancedDeltaDataset(X_val.numpy(), y_val.numpy(), augment=False)
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=0, pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Initialize models
        input_size = features.shape[1]
        self.model = EnsembleDeltaPredictor(input_size=input_size).to(self.device)
        
        # Uncertainty predictor
        self.uncertainty_model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus()
        ).to(self.device)
        
        # Loss functions
        mse_loss = nn.MSELoss()
        huber_loss = nn.HuberLoss(delta=0.1)
        
        # Optimizers
        main_optimizer = optim.AdamW(
            self.model.parameters(), lr=0.001, weight_decay=1e-5
        )
        
        uncertainty_optimizer = optim.RMSprop(
            self.uncertainty_model.parameters(), lr=0.002, alpha=0.99
        )
        
        # Schedulers
        main_scheduler = optim.lr_scheduler.OneCycleLR(
            main_optimizer, max_lr=0.01, 
            steps_per_epoch=len(train_loader), epochs=epochs,
            pct_start=0.3, anneal_strategy='cos'
        )
        
        uncertainty_scheduler = CosineAnnealingLR(
            uncertainty_optimizer, T_max=epochs//4, eta_min=1e-6
        )
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Try torch.compile if available
        try:
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("Model compiled with torch.compile")
        except:
            pass
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            self.uncertainty_model.train()
            train_loss = 0.0
            
            for batch_idx, (batch_features, batch_targets) in enumerate(train_loader):
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_targets = batch_targets.to(self.device, non_blocking=True)
                
                main_optimizer.zero_grad()
                uncertainty_optimizer.zero_grad()
                
                # Mixed precision forward pass
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        delta_pred = self.model(batch_features)
                        uncertainty_pred = self.uncertainty_model(batch_features)
                        
                        prediction_loss = 0.6 * mse_loss(delta_pred, batch_targets) + \
                                        0.4 * huber_loss(delta_pred, batch_targets)
                        uncertainty_loss = uncertainty_pred.mean() * 0.01
                        
                        total_loss = prediction_loss + uncertainty_loss
                    
                    scaler.scale(total_loss).backward()
                    
                    # Gradient clipping
                    scaler.unscale_(main_optimizer)
                    scaler.unscale_(uncertainty_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.uncertainty_model.parameters(), max_norm=1.0)
                    
                    scaler.step(main_optimizer)
                    scaler.step(uncertainty_optimizer)
                    scaler.update()
                else:
                    # Standard precision training
                    delta_pred = self.model(batch_features)
                    uncertainty_pred = self.uncertainty_model(batch_features)
                    
                    prediction_loss = 0.6 * mse_loss(delta_pred, batch_targets) + \
                                    0.4 * huber_loss(delta_pred, batch_targets)
                    uncertainty_loss = uncertainty_pred.mean() * 0.01
                    
                    total_loss = prediction_loss + uncertainty_loss
                    total_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.uncertainty_model.parameters(), max_norm=1.0)
                    
                    main_optimizer.step()
                    uncertainty_optimizer.step()
                
                main_scheduler.step()
                train_loss += total_loss.item()
            
            # Update uncertainty scheduler
            if epoch % 4 == 0:
                uncertainty_scheduler.step()
            
            # Validation phase
            self.model.eval()
            self.uncertainty_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device, non_blocking=True)
                    batch_targets = batch_targets.to(self.device, non_blocking=True)
                    
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            delta_pred = self.model(batch_features)
                            loss = mse_loss(delta_pred, batch_targets)
                    else:
                        delta_pred = self.model(batch_features)
                        loss = mse_loss(delta_pred, batch_targets)
                    
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Store history
            self.training_history['loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)