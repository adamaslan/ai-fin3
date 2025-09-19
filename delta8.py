import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

class DeltaDataset(Dataset):
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
            noise = torch.normal(0, 0.01, features.shape)
            features = features + noise
            
        return features, targets

class TransformerDeltaPredictor(nn.Module):
    """Transformer-based delta predictor with attention mechanisms"""
    def __init__(self, input_size=8, d_model=128, nhead=8, num_layers=4, dropout=0.2):
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
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Project input features to transformer dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x.unsqueeze(1) + self.pos_encoding[0:1, :]
        
        # Self-attention through transformer
        transformer_out = self.transformer(x)
        
        # Feature attention mechanism
        attended_out, _ = self.feature_attention(
            transformer_out, transformer_out, transformer_out
        )
        
        # Residual connection
        combined = transformer_out + attended_out
        
        # Pool and output
        pooled = combined.squeeze(1)
        
        return self.output_layers(pooled)

class DeltaProbabilityPredictor:
    def __init__(self, symbol='AAPL'):
        if not ALPHA_VANTAGE_API_KEY:
            raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment variables!")
        
        self.symbol = symbol.upper()
        self.api_key = ALPHA_VANTAGE_API_KEY
        self.base_url = "https://www.alphavantage.co/query"
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_price = None
        
        # Time horizons (days from this Friday)
        self.time_horizons = [7, 14, 28, 42, 56]
        
        print(f"Delta Probability Predictor - {self.symbol}")
        print(f"Device: {self.device}")
    
    def get_next_friday(self):
        """Get the date of the next Friday"""
        today = datetime.now()
        days_ahead = 4 - today.weekday()  # Friday is 4
        if days_ahead <= 0:
            days_ahead += 7
        return today + timedelta(days=days_ahead)
    
    def get_stock_data(self):
        """Fetch real stock data from Alpha Vantage"""
        print(f"Fetching real data for {self.symbol}...")
        
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': self.symbol,
            'apikey': self.api_key,
            'outputsize': 'full'
        }
        
        response = requests.get(self.base_url, params=params, timeout=30)
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            if "Note" in data:
                raise Exception("API rate limit reached. Please try again later.")
            elif "Error Message" in data:
                raise Exception(f"Invalid symbol '{self.symbol}' or API error")
            else:
                raise Exception(f"API error: {data}")
        
        # Parse data
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame(time_series).T
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Get recent data (120 days for better features)
        df_recent = df.tail(120).copy()
        self.current_price = float(df_recent['close'].iloc[-1])
        
        print(f"✓ Current {self.symbol}: ${self.current_price:.2f}")
        print(f"✓ Data points: {len(df_recent)}")
        
        return df_recent
    
    def calculate_enhanced_features(self, df):
        """Calculate enhanced technical features for delta prediction"""
        df = df.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(10).std()
        df['price_momentum'] = df['close'] / df['close'].shift(5) - 1
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['price_volume'] = df['returns'] * np.log1p(df['volume'])
        
        # Technical indicators
        df['rsi'] = self._calculate_rsi(df['close'])
        df['bb_position'] = self._bollinger_position(df['close'])
        df['sma_ratio'] = df['close'] / df['close'].rolling(20).mean()
        
        # High-low features
        df['daily_range'] = (df['high'] - df['low']) / df['close']
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        return df.fillna(method='ffill').fillna(0)
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _bollinger_position(self, prices, period=20, std_dev=2):
        """Calculate position within Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return (prices - lower_band) / (upper_band - lower_band)
    
    def create_price_targets(self):
        """Create price targets: 25 levels above and below current price"""
        targets = []
        
        # 25 price levels below current price
        for i in range(1, 26):
            target_price = self.current_price - i
            if target_price > 0:
                targets.append(target_price)
        
        # 25 price levels above current price
        for i in range(1, 26):
            target_price = self.current_price + i
            targets.append(target_price)
        
        return sorted(targets)
    
    def prepare_training_data(self, df, target_price, time_horizon):
        """Prepare training data for specific price target and time horizon"""
        feature_cols = [
            'returns', 'log_returns', 'volatility', 'price_momentum',
            'volume_ratio', 'price_volume', 'rsi', 'bb_position',
            'sma_ratio', 'daily_range', 'close_position'
        ]
        
        # Add target-specific features
        df['distance_to_target'] = (target_price - df['close']) / df['close']
        df['target_momentum'] = df['distance_to_target'].diff()
        
        feature_cols.extend(['distance_to_target', 'target_momentum'])
        
        features = df[feature_cols].dropna().values
        
        # Create sequences and labels
        sequences = []
        labels = []
        lookback = 15
        
        for i in range(lookback, len(df) - time_horizon):
            if i + lookback >= len(features):
                break
                
            sequence = features[i-lookback:i]
            
            # Future prices over time horizon
            future_prices = df['close'].iloc[i:i+time_horizon].values
            current_price_at_time = df['close'].iloc[i]
            
            # Calculate delta probability (0-50% movement toward target)
            if target_price > current_price_at_time:
                # For upward targets
                max_move = future_prices.max()
                move_percent = min(50, max((max_move - current_price_at_time) / current_price_at_time * 100, 0))
            else:
                # For downward targets
                min_move = future_prices.min()
                move_percent = min(50, max((current_price_at_time - min_move) / current_price_at_time * 100, 0))
            
            # Normalize to 0-1 range (50% = 1.0)
            delta_probability = move_percent / 50.0
            
            sequences.append(sequence.flatten())
            labels.append(delta_probability)
        
        if len(sequences) == 0:
            return None, None
            
        return np.array(sequences), np.array(labels)
    
    def train_model(self, features, targets, epochs=50, batch_size=32):
        """Train transformer model"""
        if features is None or len(features) < 20:
            return None
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data
        split_idx = int(0.8 * len(features))
        X_train, X_val = features_scaled[:split_idx], features_scaled[split_idx:]
        y_train, y_val = targets[:split_idx], targets[split_idx:]
        
        # Create datasets
        train_dataset = DeltaDataset(X_train, y_train.reshape(-1, 1), augment=True)
        val_dataset = DeltaDataset(X_val, y_val.reshape(-1, 1), augment=False)
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = TransformerDeltaPredictor(input_size=features.shape[1]).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        model.load_state_dict(best_model)
        return model
    
    def predict_delta_probability(self, model, latest_features):
        """Predict delta probability for latest data"""
        if model is None:
            return 25.0  # Default 25% if no model
        
        model.eval()
        with torch.no_grad():
            features_tensor = torch.from_numpy(latest_features.astype(np.float32)).unsqueeze(0).to(self.device)
            probability = model(features_tensor).cpu().numpy()[0][0]
            return min(50.0, max(0.0, probability * 50.0))  # Scale to 0-50%
    
    def run_analysis(self):
        """Run complete delta probability analysis"""
        try:
            # Get data
            df = self.get_stock_data()
            df = self.calculate_enhanced_features(df)
            
            # Create price targets
            price_targets = self.create_price_targets()
            
            # Get next Friday date
            next_friday = self.get_next_friday()
            
            print(f"\nAnalyzing {len(price_targets)} price targets across {len(self.time_horizons)} time horizons...")
            print(f"Next Friday: {next_friday.strftime('%Y-%m-%d')}")
            
            results = {}
            
            for time_horizon in self.time_horizons:
                target_date = next_friday + timedelta(days=time_horizon)
                results[time_horizon] = {
                    'date': target_date,
                    'targets': {}
                }
                
                print(f"\nProcessing {time_horizon}-day horizon ({target_date.strftime('%Y-%m-%d')})...")
                
                for i, target_price in enumerate(price_targets):
                    # Prepare training data
                    features, targets = self.prepare_training_data(df, target_price, time_horizon)
                    
                    if features is not None:
                        # Train model
                        model = self.train_model(features, targets)
                        
                        # Get prediction for latest data
                        latest_features = self.scaler.transform(features[-1:])
                        delta_prob = self.predict_delta_probability(model, latest_features[0])
                    else:
                        delta_prob = 25.0
                    
                    results[time_horizon]['targets'][target_price] = {
                        'delta_probability': delta_prob,
                        'price_change': (target_price - self.current_price) / self.current_price * 100,
                        'distance': abs(target_price - self.current_price)
                    }
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Completed {i + 1}/{len(price_targets)} targets")
            
            return results
            
        except Exception as e:
            print(f"ERROR: {e}")
            return None
    
    def display_results(self, results):
        """Display results in formatted tables"""
        if not results:
            return
        
        print(f"\n{'='*80}")
        print(f"DELTA PROBABILITY ANALYSIS - {self.symbol}")
        print(f"Current Price: ${self.current_price:.2f}")
        print(f"{'='*80}")
        
        for time_horizon in self.time_horizons:
            data = results[time_horizon]
            print(f"\n📅 TIME HORIZON: {time_horizon} DAYS ({data['date'].strftime('%Y-%m-%d')})")
            print(f"{'Price':>8} {'Change%':>8} {'Delta%':>8} {'Distance':>10}")
            print("-" * 36)
            
            # Sort by delta probability (descending)
            sorted_targets = sorted(data['targets'].items(), 
                                  key=lambda x: x[1]['delta_probability'], reverse=True)
            
            # Show top 10 targets
            for target_price, target_data in sorted_targets[:10]:
                print(f"${target_price:>7.2f} {target_data['price_change']:>7.1f}% "
                      f"{target_data['delta_probability']:>7.1f}% ${target_data['distance']:>9.2f}")
            
            # Find best upward and downward targets
            upward_targets = [(p, d) for p, d in data['targets'].items() if p > self.current_price]
            downward_targets = [(p, d) for p, d in data['targets'].items() if p < self.current_price]
            
            if upward_targets:
                best_up = max(upward_targets, key=lambda x: x[1]['delta_probability'])
                print(f"\n🚀 Best Upward:  ${best_up[0]:.2f} ({best_up[1]['delta_probability']:.1f}%)")
            
            if downward_targets:
                best_down = max(downward_targets, key=lambda x: x[1]['delta_probability'])
                print(f"📉 Best Downward: ${best_down[0]:.2f} ({best_down[1]['delta_probability']:.1f}%)")
    
    def create_visualization(self, results):
        """Create visualization of delta probabilities"""
        if not results:
            return
        
        fig, axes = plt.subplots(len(self.time_horizons), 1, figsize=(12, 4*len(self.time_horizons)))
        if len(self.time_horizons) == 1:
            axes = [axes]
        
        for i, time_horizon in enumerate(self.time_horizons):
            data = results[time_horizon]
            
            prices = list(data['targets'].keys())
            probabilities = [data['targets'][p]['delta_probability'] for p in prices]
            colors = ['red' if p < self.current_price else 'green' for p in prices]
            
            axes[i].bar(range(len(prices)), probabilities, color=colors, alpha=0.7)
            axes[i].axvline(x=25, color='black', linestyle='--', alpha=0.8, linewidth=2)  # Current price line
            axes[i].set_title(f'{self.symbol} - Delta Probabilities ({time_horizon} days)')
            axes[i].set_ylabel('Delta Probability (%)')
            axes[i].set_ylim(0, 50)
            axes[i].grid(True, alpha=0.3)
            
            # Set x-axis labels (every 5th price)
            tick_positions = range(0, len(prices), 5)
            tick_labels = [f"${prices[i]:.0f}" for i in tick_positions]
            axes[i].set_xticks(tick_positions)
            axes[i].set_xticklabels(tick_labels, rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.symbol}_delta_probabilities.png', dpi=300, bbox_inches='tight')
        plt.show()

# USAGE EXAMPLE
if __name__ == "__main__":
    # Create predictor for AAPL
    predictor = DeltaProbabilityPredictor(symbol='AAPL')
    
    # Run analysis
    results = predictor.run_analysis()
    
    if results:
        # Display results
        predictor.display_results(results)
        
        # Create visualization
        predictor.create_visualization(results)
        
        print("\n✅ Analysis completed successfully!")
    else:
        print("❌ Analysis failed. Check API key and connection.")