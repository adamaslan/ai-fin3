#delta 3 shortened 

# most pytorch elements
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import RobustScaler
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
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
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
        calls['mid_price'] = (calls['bid'] + calls['ask']) / 2
        calls['log_volume'] = np.log1p(calls['volume'])
        
        # Time-scaled features
        calls['moneyness_time'] = calls['moneyness'] * calls['sqrt_time'] 
        calls['iv_time'] = calls['implied_volatility'] * calls['sqrt_time']
        
        # Define feature columns
        feature_columns = [
            'log_moneyness',
            'sqrt_time', 
            'implied_volatility',
            'log_volume',
            'moneyness_time',
            'iv_time'
        ]
        
        # Clean data
        calls = calls.dropna(subset=feature_columns + ['delta'])
        calls = calls[np.isfinite(calls[feature_columns + ['delta']]).all(axis=1)]
        
        features = calls[feature_columns].values
        targets = calls['delta'].values.reshape(-1, 1)
        
        print(f"Training data prepared: {len(features)} samples, {features.shape[1]} features")
        
        return features, targets
    
    def train_model(self, features, targets, epochs=50, batch_size=64, patience=10):
        """Train LSTM model for delta prediction"""
        print(f"Training LSTM model ({epochs} epochs, batch_size={batch_size})...")
        
        # Data scaling
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data
        indices = torch.randperm(len(features_scaled))
        split_idx = int(0.8 * len(features_scaled))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        X_train = features_scaled[train_indices]
        y_train = targets[train_indices]
        X_val = features_scaled[val_indices]
        y_val = targets[val_indices]
        
        # Create datasets
        train_dataset = EnhancedDeltaDataset(X_train, y_train, augment=True)
        val_dataset = EnhancedDeltaDataset(X_val, y_val, augment=False)
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=0, pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Initialize model
        input_size = features.shape[1]
        self.model = LSTMDeltaPredictor(input_size=input_size).to(self.device)
        
        # Loss function
        loss_fn = nn.MSELoss()
        
        # Optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                delta_pred = self.model(batch_features)
                loss = loss_fn(delta_pred, batch_targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    delta_pred = self.model(batch_features)
                    loss = loss_fn(delta_pred, batch_targets)
                    
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Store history
            self.training_history['loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Update learning rate
            scheduler.step()
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_delta_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_delta_model.pth'))
        self.model_trained = True
        
        print(f"Model training completed. Best validation loss: {best_val_loss:.6f}")
        return self.training_history
    
    def predict_delta(self, option_features):
        """Predict delta for given option features"""
        if not self.model_trained:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Ensure features are in the right format
        if isinstance(option_features, pd.DataFrame):
            feature_columns = [
                'log_moneyness', 'sqrt_time', 'implied_volatility', 
                'log_volume', 'moneyness_time', 'iv_time'
            ]
            features = option_features[feature_columns].values
        else:
            features = option_features
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        features_tensor = torch.from_numpy(features_scaled.astype(np.float32)).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            delta_pred = self.model(features_tensor)
        
        return delta_pred.cpu().numpy()
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['loss'], label='Training Loss')
        plt.plot(self.training_history['val_loss'], label='Validation Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        
        plt.tight_layout()
        plt.savefig('delta_training_history.png')
        plt.close()
        
        print("Training history plot saved to 'delta_training_history.png'")
    
    def run_delta_analysis(self, symbol='AAPL'):
        """Run complete delta analysis pipeline"""
        try:
            # Get data
            options_df = self.get_sample_data()
            
            # Prepare training data
            features, targets = self.prepare_enhanced_training_data(options_df)
            
            # Train model
            self.train_model(features, targets)
            
            # Plot training history
            self.plot_training_history()
            
            # Make predictions on the original data
            predictions = self.predict_delta(features)
            
            # Calculate metrics
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            
            print(f"Model Performance - MSE: {mse:.6f}, MAE: {mae:.6f}")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_df = pd.DataFrame({
                'Actual_Delta': targets.flatten(),
                'Predicted_Delta': predictions.flatten(),
                'Absolute_Error': np.abs(predictions.flatten() - targets.flatten())
            })
            
            result_df.to_csv(f"{symbol}_delta_analysis_{timestamp}.csv", index=False)
            print(f"Results saved to {symbol}_delta_analysis_{timestamp}.csv")
            
            return result_df
            
        except Exception as e:
            print(f"Error in delta analysis: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    predictor = EnhancedLiveDeltaPredictor(api_key=ALPHA_VANTAGE_API_KEY)
    results = predictor.run_delta_analysis()