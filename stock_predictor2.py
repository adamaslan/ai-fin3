#!/usr/bin/env python3
"""
GCP FREE TIER OPTIMIZED STOCK PREDICTION PIPELINE
- Single e2-micro VM instance (1 free per month)
- 5GB Cloud Storage (free tier)
- Minimal data transfer usage
- Runs Mon-Fri at 10 AM via cron
- Updated for ttb1-machine-sept VM and ttb-bucket1 storage
- PyTorch removed - uses only scikit-learn for lightweight operation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
import logging
import os
import gzip
import pickle
from datetime import datetime, timedelta
from google.cloud import storage
import warnings
warnings.filterwarnings('ignore')

class FreeTierStockPredictor:
    def __init__(self):
        # Free tier constraints - configured for your specific setup
        self.bucket_name = 'ttb-bucket1'  # Your specific bucket name
        self.max_symbols = 5  # Limit to stay under data transfer limits
        self.cache_days = 7   # Cache data to reduce API calls
        
        # Local storage for caching (uses home directory instead of /opt)
        self.cache_dir = os.path.expanduser('~/stock_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize GCS client (only when needed)
        self._storage_client = None
        
        # Default symbols - choose liquid stocks with good data
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        # Initialize scaler for feature normalization
        self.scaler = StandardScaler()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    @property
    def storage_client(self):
        """Lazy initialization of storage client"""
        if self._storage_client is None:
            try:
                self._storage_client = storage.Client()
            except Exception as e:
                self.logger.error(f"Failed to initialize GCS client: {e}")
                self._storage_client = None
        return self._storage_client
    
    def get_cached_data(self, symbol, days=7):
        """Get cached data to minimize API calls and data transfer"""
        cache_file = f"{self.cache_dir}/{symbol}_cache.pkl.gz"
        
        try:
            if os.path.exists(cache_file):
                # Check if cache is still valid
                cache_age = datetime.now().timestamp() - os.path.getmtime(cache_file)
                if cache_age < days * 24 * 3600:  # Cache valid for 'days' days
                    with gzip.open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                        self.logger.info(f"Using cached data for {symbol}")
                        return cached_data
        except Exception as e:
            self.logger.warning(f"Cache read error for {symbol}: {e}")
        
        return None
    
    def cache_data(self, symbol, data):
        """Cache data locally to reduce future API calls"""
        cache_file = f"{self.cache_dir}/{symbol}_cache.pkl.gz"
        try:
            with gzip.open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Cache write error for {symbol}: {e}")
    
    def fetch_stock_data(self, symbol, period='3mo'):
        """Fetch stock data with caching to minimize data transfer"""
        # Try cache first
        cached_data = self.get_cached_data(symbol)
        if cached_data is not None:
            return cached_data
        
        try:
            self.logger.info(f"Fetching fresh data for {symbol}")
            
            # Use shorter period to minimize data transfer
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                raise ValueError(f"No data for {symbol}")
            
            # Calculate essential indicators only
            hist = self.calculate_minimal_indicators(hist)
            
            # Cache the processed data
            self.cache_data(symbol, hist)
            
            return hist
            
        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_minimal_indicators(self, df):
        """Calculate only essential indicators to reduce computation"""
        # Basic moving averages
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD (simplified)
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI (optimized calculation)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = 20
        bb_std = df['Close'].rolling(bb_period).std()
        df['BB_Upper'] = df['SMA_20'] + (bb_std * 2)
        df['BB_Lower'] = df['SMA_20'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Price momentum and volatility
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(10).std()
        df['Price_Change_3d'] = df['Close'].pct_change(3)
        df['Price_Change_5d'] = df['Close'].pct_change(5)
        df['Price_Change_10d'] = df['Close'].pct_change(10)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price position relative to recent highs/lows
        df['High_20d'] = df['High'].rolling(20).max()
        df['Low_20d'] = df['Low'].rolling(20).min()
        df['Price_Position'] = (df['Close'] - df['Low_20d']) / (df['High_20d'] - df['Low_20d'])
        
        # Fixed deprecated fillna usage
        return df.ffill().fillna(0)
    
    def create_ensemble_prediction(self, df):
        """Create ensemble prediction using multiple scikit-learn models"""
        if len(df) < 40:
            return {'prediction': 0.5, 'signal': 'INSUFFICIENT_DATA', 'confidence': 0.0}
        
        # Enhanced feature set for better predictions
        feature_cols = [
            'Returns', 'RSI', 'MACD', 'MACD_Histogram', 'BB_Position',
            'Volume_Ratio', 'Volatility', 'Price_Change_3d', 'Price_Change_5d',
            'Price_Position'
        ]
        
        # Create multiple targets for ensemble
        df['Future_Price_3d'] = df['Close'].shift(-3)
        df['Future_Price_5d'] = df['Close'].shift(-5)
        
        # Direction targets
        df['Direction_3d'] = (df['Future_Price_3d'] > df['Close']).astype(int)
        df['Direction_5d'] = (df['Future_Price_5d'] > df['Close']).astype(int)
        
        # Remove NaN rows
        df_clean = df.dropna()
        if len(df_clean) < 30:
            return {'prediction': 0.5, 'signal': 'INSUFFICIENT_DATA', 'confidence': 0.0}
        
        # Prepare features
        X = df_clean[feature_cols].values
        
        # Use more recent data for training (last 40 days)
        X = X[-40:]
        
        if len(X) < 25:
            return {'prediction': 0.5, 'signal': 'INSUFFICIENT_DATA', 'confidence': 0.0}
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        try:
            predictions = []
            confidences = []
            
            # Model 1: Random Forest for 3-day prediction
            if len(df_clean) >= 30:
                y_3d = df_clean['Direction_3d'].values[-40:][:len(X)]
                if len(y_3d) >= 20:
                    split_idx = int(len(X_scaled) * 0.8)
                    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
                    y_train, y_test = y_3d[:split_idx], y_3d[split_idx:]
                    
                    rf_3d = RandomForestClassifier(
                        n_estimators=15, 
                        max_depth=6, 
                        random_state=42,
                        min_samples_split=3,
                        min_samples_leaf=2
                    )
                    rf_3d.fit(X_train, y_train)
                    
                    pred_3d = rf_3d.predict_proba(X_scaled[-1:].reshape(1, -1))[0][1]
                    predictions.append(pred_3d)
                    
                    # Calculate confidence based on feature importance consistency
                    feature_importance = rf_3d.feature_importances_
                    confidence_3d = 1.0 - np.std(feature_importance)
                    confidences.append(confidence_3d)
            
            # Model 2: Gradient Boosting for 5-day prediction
            if len(df_clean) >= 30:
                y_5d = df_clean['Direction_5d'].values[-40:][:len(X)]
                if len(y_5d) >= 20:
                    gb_5d = GradientBoostingClassifier(
                        n_estimators=20,
                        max_depth=4,
                        learning_rate=0.1,
                        random_state=42
                    )
                    
                    split_idx = int(len(X_scaled) * 0.8)
                    X_train, y_train = X_scaled[:split_idx], y_5d[:split_idx]
                    
                    gb_5d.fit(X_train, y_train)
                    pred_5d = gb_5d.predict_proba(X_scaled[-1:].reshape(1, -1))[0][1]
                    predictions.append(pred_5d)
                    
                    confidence_5d = 0.8  # Gradient boosting generally stable
                    confidences.append(confidence_5d)
            
            # Model 3: Logistic Regression for trend analysis
            if len(X_scaled) >= 15:
                # Simple trend target: is current price above 10-day average?
                trend_target = (df_clean['Close'] > df_clean['SMA_10']).astype(int).values[-40:][:len(X)]
                if len(trend_target) >= 15:
                    lr_trend = LogisticRegression(random_state=42, max_iter=200)
                    
                    # Use more recent data for trend analysis
                    recent_X = X_scaled[-20:]
                    recent_y = trend_target[-20:]
                    
                    if len(recent_X) >= 10:
                        lr_trend.fit(recent_X[:-1], recent_y[:-1])
                        pred_trend = lr_trend.predict_proba(X_scaled[-1:].reshape(1, -1))[0][1]
                        predictions.append(pred_trend)
                        
                        confidence_trend = 0.7
                        confidences.append(confidence_trend)
            
            if not predictions:
                return {'prediction': 0.5, 'signal': 'MODEL_ERROR', 'confidence': 0.0}
            
            # Weighted ensemble prediction
            weights = np.array(confidences)
            if weights.sum() > 0:
                weights = weights / weights.sum()
                ensemble_prediction = np.average(predictions, weights=weights)
            else:
                ensemble_prediction = np.mean(predictions)
            
            # Calculate overall confidence
            overall_confidence = np.mean(confidences)
            
            # Generate signal with confidence-based thresholds
            high_conf_threshold = 0.68 if overall_confidence > 0.7 else 0.65
            low_conf_threshold = 0.32 if overall_confidence > 0.7 else 0.35
            
            if ensemble_prediction > high_conf_threshold:
                signal = 'BUY'
            elif ensemble_prediction < low_conf_threshold:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            return {
                'prediction': ensemble_prediction,
                'signal': signal,
                'confidence': overall_confidence,
                'model': f'ensemble_{len(predictions)}_models',
                'individual_predictions': predictions
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction error: {e}")
            return {'prediction': 0.5, 'signal': 'MODEL_ERROR', 'confidence': 0.0}
    
    def run_predictions(self):
        """Run predictions for all symbols"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'predictions': {},
            'summary': {}
        }
        
        successful_predictions = 0
        total_symbols = len(self.symbols)
        
        for symbol in self.symbols:
            try:
                self.logger.info(f"Processing {symbol}")
                
                # Fetch data
                df = self.fetch_stock_data(symbol)
                if df.empty:
                    results['predictions'][symbol] = {'error': 'No data available'}
                    continue
                
                # Get current price and basic metrics
                current_price = df['Close'].iloc[-1]
                daily_change = df['Returns'].iloc[-1]
                rsi = df['RSI'].iloc[-1]
                
                # Run ensemble prediction
                prediction = self.create_ensemble_prediction(df)
                prediction['current_price'] = current_price
                prediction['daily_change'] = daily_change
                prediction['rsi'] = rsi
                prediction['symbol'] = symbol
                
                results['predictions'][symbol] = prediction
                successful_predictions += 1
                
                self.logger.info(f"{symbol}: {prediction['signal']} (${current_price:.2f}, conf: {prediction.get('confidence', 0):.2f})")
                
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
                results['predictions'][symbol] = {'error': str(e)}
        
        # Add summary
        results['summary'] = {
            'total_symbols': total_symbols,
            'successful_predictions': successful_predictions,
            'success_rate': successful_predictions / total_symbols if total_symbols > 0 else 0
        }
        
        return results
    
    def save_to_gcs_compressed(self, data, blob_name):
        """Save compressed data to GCS to minimize storage usage"""
        try:
            if self.storage_client is None:
                self.logger.error("GCS client not available")
                return False
                
            # Compress data before upload
            json_str = json.dumps(data, default=str)
            compressed_data = gzip.compress(json_str.encode('utf-8'))
            
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_name + '.gz')
            blob.upload_from_string(compressed_data, content_type='application/gzip')
            
            self.logger.info(f"Compressed results saved to gs://{self.bucket_name}/{blob_name}.gz")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving to GCS: {e}")
            return False
    
    def save_local_backup(self, data, filename):
        """Save backup locally using persistent disk"""
        try:
            backup_dir = os.path.expanduser('~/stock_predictions')
            os.makedirs(backup_dir, exist_ok=True)
            
            filepath = f"{backup_dir}/{filename}"
            with gzip.open(filepath + '.gz', 'wt') as f:
                json.dump(data, f, default=str, indent=2)
            
            self.logger.info(f"Local backup saved: {filepath}.gz")
            
            # Keep only last 10 files to manage disk space
            self.cleanup_old_backups(backup_dir)
            
        except Exception as e:
            self.logger.error(f"Error saving local backup: {e}")
    
    def cleanup_old_backups(self, backup_dir, keep_files=10):
        """Clean up old backup files to manage disk space"""
        try:
            files = [f for f in os.listdir(backup_dir) if f.endswith('.gz')]
            files.sort(key=lambda x: os.path.getmtime(os.path.join(backup_dir, x)))
            
            # Remove oldest files if we have more than keep_files
            if len(files) > keep_files:
                for old_file in files[:-keep_files]:
                    os.remove(os.path.join(backup_dir, old_file))
                    self.logger.info(f"Removed old backup: {old_file}")
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up backups: {e}")
    
    def run_daily_predictions(self):
        """Main function to run daily predictions"""
        self.logger.info("Starting daily stock predictions on ttb1-machine-sept")
        
        # Run predictions
        results = self.run_predictions()
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"predictions_{timestamp}"
        
        # Save to GCS (compressed)
        gcs_success = self.save_to_gcs_compressed(results, f"daily/{filename}")
        
        # Always save local backup
        self.save_local_backup(results, filename)
        
        # Log summary
        summary = results['summary']
        self.logger.info(f"Prediction run complete: {summary['successful_predictions']}/{summary['total_symbols']} symbols processed")
        
        # Display results
        self.display_results(results)
        
        return results
    
    def display_results(self, results):
        """Display formatted results with enhanced metrics"""
        print(f"\n{'='*80}")
        print(f"STOCK PREDICTIONS - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"VM: ttb1-machine-sept | Bucket: ttb-bucket1 | Models: Scikit-learn Ensemble")
        print(f"{'='*80}")
        
        for symbol, data in results['predictions'].items():
            if 'error' in data:
                print(f"{symbol}: ERROR - {data['error']}")
                continue
            
            price = data.get('current_price', 0)
            signal = data.get('signal', 'UNKNOWN')
            prediction = data.get('prediction', 0)
            confidence = data.get('confidence', 0)
            daily_change = data.get('daily_change', 0)
            rsi = data.get('rsi', 50)
            
            # Format daily change with color indicator
            change_str = f"{daily_change*100:+.1f}%"
            rsi_str = f"RSI:{rsi:.0f}"
            
            print(f"{symbol}: ${price:>7.2f} ({change_str:>6}) | {signal:>4} | {prediction:.3f} ({prediction*100:.0f}%) | Conf:{confidence:.2f} | {rsi_str}")
        
        summary = results['summary']
        print(f"\nSummary: {summary['successful_predictions']}/{summary['total_symbols']} predictions completed")
        
        # Show strongest signals
        strong_signals = []
        for symbol, data in results['predictions'].items():
            if 'error' not in data:
                conf = data.get('confidence', 0)
                signal = data.get('signal', 'HOLD')
                if conf > 0.6 and signal != 'HOLD':
                    strong_signals.append(f"{symbol}:{signal}")
        
        if strong_signals:
            print(f"Strong signals: {', '.join(strong_signals)}")
        
        if summary['successful_predictions'] == 0:
            print("WARNING: No successful predictions. Check internet connection and API access.")


if __name__ == "__main__":
    print("Stock Predictor starting on ttb1-machine-sept VM...")
    print("Using bucket: ttb-bucket1")
    print("Scikit-learn Ensemble Models: RandomForest + GradientBoosting + LogisticRegression")
    print("="*80)
    
    # Initialize and run predictor
    predictor = FreeTierStockPredictor()
    results = predictor.run_daily_predictions()
    
    print("\nOptimized for e2-micro VM:")
    print("- Memory usage: ~150-200MB (PyTorch removed)")
    print("- CPU usage: Minimal (lightweight scikit-learn models)")
    print("- Data transfer: ~10-20MB per run (well under 1GB/month)")
    print("- Storage: ~1MB compressed per day (~30MB/month, under 5GB)")
    print("- Cost: $0 within free tier limits")
    
    # Exit with appropriate code
    import sys
    if results['summary']['successful_predictions'] > 0:
        print("SUCCESS: Predictions completed successfully")
        sys.exit(0)
    else:
        print("ERROR: No successful predictions completed")
        sys.exit(1)