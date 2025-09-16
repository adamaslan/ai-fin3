#!/usr/bin/env python3
"""
GCP FREE TIER OPTIMIZED STOCK PREDICTION PIPELINE
- Single e2-micro VM instance (1 free per month)
- 5GB Cloud Storage (free tier)
- Minimal data transfer usage
- Runs Mon-Fri at 10 AM via cron
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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
        # Free tier constraints
        self.bucket_name = os.environ.get('GCS_BUCKET', 'stock-predictions-free')
        self.max_symbols = 5  # Limit to stay under data transfer limits
        self.cache_days = 7   # Cache data to reduce API calls
        
        # Local storage for caching (uses persistent disk)
        self.cache_dir = '/opt/stock_cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize GCS client (only when needed)
        self._storage_client = None
        
        # Default symbols - choose liquid stocks with good data
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    @property
    def storage_client(self):
        """Lazy initialization of storage client"""
        if self._storage_client is None:
            self._storage_client = storage.Client()
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
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD (simplified)
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # RSI (simplified calculation)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Price momentum
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(10).std()
        df['Price_Change_5d'] = df['Close'].pct_change(5)
        
        # Volume ratio
        df['Volume_SMA'] = df['Volume'].rolling(10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df.fillna(method='ffill').fillna(0)
    
    def create_lightweight_prediction(self, df):
        """Create fast, lightweight prediction using minimal features"""
        if len(df) < 30:
            return {'prediction': 0.5, 'signal': 'INSUFFICIENT_DATA'}
        
        # Select minimal feature set
        feature_cols = ['Returns', 'RSI', 'MACD', 'Volume_Ratio', 
                       'Volatility', 'Price_Change_5d']
        
        # Create simple target: price direction in 3 days
        df['Future_Price'] = df['Close'].shift(-3)
        df['Direction'] = (df['Future_Price'] > df['Close']).astype(int)
        
        # Remove NaN rows
        df_clean = df.dropna()
        if len(df_clean) < 20:
            return {'prediction': 0.5, 'signal': 'INSUFFICIENT_DATA'}
        
        # Prepare data
        X = df_clean[feature_cols].values
        y = df_clean['Direction'].values
        
        # Use only the most recent data for training (reduce computation)
        X = X[-30:]  # Last 30 days only
        y = y[-30:]
        
        if len(X) < 15:
            return {'prediction': 0.5, 'signal': 'INSUFFICIENT_DATA'}
        
        # Simple train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Use lightweight Random Forest (fewer estimators)
        model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate accuracy if we have test data
        accuracy = 0.5
        if len(X_test) > 0:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
        
        # Make prediction on latest data
        latest_features = X[-1:].reshape(1, -1)
        prediction_proba = model.predict_proba(latest_features)[0][1]
        
        # Generate signal
        if prediction_proba > 0.65:
            signal = 'BUY'
        elif prediction_proba < 0.35:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            'prediction': prediction_proba,
            'signal': signal,
            'accuracy': accuracy,
            'model': 'lightweight_rf'
        }
    
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
                
                # Get current price
                current_price = df['Close'].iloc[-1]
                
                # Run prediction
                prediction = self.create_lightweight_prediction(df)
                prediction['current_price'] = current_price
                prediction['symbol'] = symbol
                
                results['predictions'][symbol] = prediction
                successful_predictions += 1
                
                self.logger.info(f"{symbol}: {prediction['signal']} (${current_price:.2f})")
                
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
            backup_dir = '/opt/stock_predictions'
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
        self.logger.info("Starting daily stock predictions")
        
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
        """Display formatted results"""
        print(f"\n{'='*60}")
        print(f"STOCK PREDICTIONS - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")
        
        for symbol, data in results['predictions'].items():
            if 'error' in data:
                print(f"{symbol}: ERROR - {data['error']}")
                continue
            
            price = data.get('current_price', 0)
            signal = data.get('signal', 'UNKNOWN')
            prediction = data.get('prediction', 0)
            accuracy = data.get('accuracy', 0)
            
            print(f"{symbol}: ${price:>7.2f} | {signal:>4} | {prediction:.3f} ({prediction*100:.1f}%) | Acc: {accuracy:.2f}")
        
        summary = results['summary']
        print(f"\nSummary: {summary['successful_predictions']}/{summary['total_symbols']} predictions completed")


# Setup script for GCP VM
def setup_gcp_environment():
    """Setup script to run on GCP VM"""
    setup_commands = '''
# GCP VM Setup Script
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# Create project directory
mkdir -p /opt/stock-predictor
cd /opt/stock-predictor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install yfinance pandas numpy torch scikit-learn google-cloud-storage

# Setup service account (download key from GCP Console)
export GOOGLE_APPLICATION_CREDENTIALS="/opt/stock-predictor/service-account-key.json"

# Create cron job for weekdays at 10 AM EST
# Add to crontab: crontab -e
# 0 15 * * 1-5 /opt/stock-predictor/venv/bin/python /opt/stock-predictor/main.py >> /var/log/stock-predictions.log 2>&1
'''
    return setup_commands


# Deployment and configuration
class GCPDeployment:
    def __init__(self):
        self.project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
        self.bucket_name = 'stock-predictions-free'
        self.vm_zone = 'us-central1-a'  # Free tier region
    
    def create_storage_bucket(self):
        """Create GCS bucket in free tier region"""
        create_bucket_command = f'''
gcloud storage buckets create gs://{self.bucket_name} \
    --location=us-central1 \
    --uniform-bucket-level-access
'''
        return create_bucket_command
    
    def create_vm_instance(self):
        """Create e2-micro VM instance"""
        create_vm_command = f'''
gcloud compute instances create stock-predictor-vm \
    --zone={self.vm_zone} \
    --machine-type=e2-micro \
    --boot-disk-size=30GB \
    --boot-disk-type=pd-standard \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --scopes=storage-rw \
    --tags=stock-predictor
'''
        return create_vm_command
    
    def get_startup_script(self):
        """Get startup script for VM"""
        return '''#!/bin/bash
# Startup script for stock predictor VM
cd /opt/stock-predictor
source venv/bin/activate
export GOOGLE_APPLICATION_CREDENTIALS="/opt/stock-predictor/service-account-key.json"
python main.py
'''


if __name__ == "__main__":
    # Initialize and run predictor
    predictor = FreeTierStockPredictor()
    results = predictor.run_daily_predictions()
    
    print("\nFree tier usage estimate:")
    print("- Data transfer: ~10-20MB per run (well under 1GB/month)")
    print("- Storage: ~1MB compressed per day (~30MB/month, under 5GB)")
    print("- Compute: e2-micro instance (1 free per month)")
    print("- Cost: $0 if staying within free tier limits")