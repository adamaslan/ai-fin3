import requests
import json
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import date, datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Remove TensorFlow imports
# Add PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
API_KEY = ALPHA_VANTAGE_API_KEY
SYMBOL = 'AAPL' # Apple Inc.

# API Parameters
# Modified to target a specific 48-hour period
# Note: We'll need to fetch two days of data
today = datetime.now()
end_date = today
start_date = today - timedelta(days=30)  # Increase from 2 to 30 days or more

# Format dates for API and filtering
end_date_str = end_date.strftime('%Y-%m-%d')
start_date_str = start_date.strftime('%Y-%m-%d')
month_str = start_date.strftime('%Y-%m')

# Parameters for 5-minute interval
params_5min = {
    "function": "BBANDS",
    "symbol": SYMBOL,
    "interval": "5min",        # 5-minute intervals
    "month": month_str,        # Month containing our start date
    "time_period": "20",
    "series_type": "close",
    "apikey": API_KEY,
    "datatype": "json",
    "outputsize": "full"
}

# Parameters for 1-hour interval
params_1hour = {
    "function": "BBANDS",
    "symbol": SYMBOL,
    "interval": "60min",       # 1-hour intervals
    "month": month_str,        # Month containing our start date
    "time_period": "20",
    "series_type": "close",
    "apikey": API_KEY,
    "datatype": "json",
    "outputsize": "full"
}

# Alpha Vantage API endpoint
API_URL = "https://www.alphavantage.co/query"

# --- Autoencoder Helper Functions ---
def create_sequences(data, sequence_length):
    """
    Creates sequences from the input data using PyTorch's unfold operation.
    data: pandas DataFrame with features
    sequence_length: Number of time steps in each sequence
    """
    # Convert DataFrame to numpy array
    data_array = data.values
    
    # Convert to PyTorch tensor
    data_tensor = torch.FloatTensor(data_array)
    
    # Create sequences using unfold
    sequences = []
    for i in range(data_tensor.shape[1]):  # For each feature
        feature_sequences = data_tensor[:, i].unfold(0, sequence_length, 1)
        sequences.append(feature_sequences)
    
    # Stack features
    result = torch.stack(sequences, dim=2)
    
    return result.numpy()

# Replace TensorFlow autoencoder with PyTorch implementation
class BollingerAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim_factor=0.5):
        super(BollingerAutoencoder, self).__init__()
        
        # Calculate dimensions for the bottleneck
        hidden_dim = int(input_dim * 0.75)
        bottleneck_dim = int(input_dim * encoding_dim_factor)
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU()
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Sigmoid for 0-1 normalized data
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def define_autoencoder(input_shape, encoding_dim_factor=0.5):
    """
    Defines a PyTorch autoencoder model.
    input_shape: tuple, shape of the input data
    encoding_dim_factor: determines the size of the bottleneck layer
    """
    if len(input_shape) == 2:  # (sequence_length, num_features)
        flat_input_dim = input_shape[0] * input_shape[1]
    else:  # (num_features,)
        flat_input_dim = input_shape[0]
    
    model = BollingerAutoencoder(flat_input_dim, encoding_dim_factor)
    return model

def get_reconstruction_error(data_scaled, model):
    """Calculates reconstruction error using PyTorch."""
    # Convert to PyTorch tensor
    data_tensor = torch.FloatTensor(data_scaled)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        predictions = model(data_tensor)
    
    # Calculate MSE
    mse = torch.mean((data_tensor - predictions) ** 2, dim=1).numpy()
    return mse

# Remove the TensorFlow version of get_reconstruction_error

# Add device selection at the beginning of the script
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Remove this out-of-place code:
# Then modify the model training to use the selected device
# model = define_autoencoder(input_shape).to(device)
# train_tensor = torch.FloatTensor(train_data).to(device)
# test_tensor = torch.FloatTensor(test_data).to(device)
# In the anomaly detection visualization section:
def plot_anomalies(df_filtered, reconstruction_errors, threshold, interval_name):
    """Plot the anomalies detected by the autoencoder."""
    plt.figure(figsize=(12, 8))
    
    # Convert to PyTorch tensors for easier manipulation
    errors_tensor = torch.FloatTensor(reconstruction_errors)
    
    # Plot the reconstruction error
    plt.subplot(2, 1, 1)
    plt.plot(errors_tensor.numpy())
    plt.axhline(y=threshold, color='r', linestyle='--')
    plt.title(f'Reconstruction Error ({interval_name})')
    plt.ylabel('Error')
    plt.grid(True, alpha=0.3)
    
    # Plot the Bollinger Bands with anomalies highlighted
    plt.subplot(2, 1, 2)
    plt.plot(df_filtered.index, df_filtered['Real Middle Band'], 'b-', label='Middle Band')
    plt.plot(df_filtered.index, df_filtered['Real Upper Band'], 'g-', label='Upper Band')
    plt.plot(df_filtered.index, df_filtered['Real Lower Band'], 'r-', label='Lower Band')
    
    # Highlight anomalies
    anomalies = errors_tensor > threshold
    anomaly_indices = torch.nonzero(anomalies).squeeze().numpy()
    
    if len(anomaly_indices) > 0:
        plt.scatter(
            df_filtered.index[anomaly_indices], 
            df_filtered['Real Middle Band'].iloc[anomaly_indices],
            color='purple', s=50, zorder=5, label='Anomalies'
        )
    
    plt.title(f'Bollinger Bands with Anomalies ({interval_name})')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'{SYMBOL}_anomalies_{interval_name}.png')
    print(f"\nAnomaly detection chart saved as {SYMBOL}_anomalies_{interval_name}.png")
    plt.show()


def fetch_and_process_bbands(params, interval_name):
    """
    Fetch data from Alpha Vantage API and process it.
    params: dictionary with Alpha Vantage API parameters
    interval_name: name of the time interval
    """
    response = requests.get(API_URL, params=params)
    
    # Convert to pandas DataFrame
    df = pd.DataFrame.from_dict(response.json()['values'])
    
    # Convert to float
    df = df.astype(float)
    
    return df

def visualize_bollinger_bands(df_filtered, interval_name):
    """
    Visualization of Bollinger Bands data.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot the Bollinger Bands
    plt.plot(df_filtered.index, df_filtered['Real Middle Band'], 'b-', label='Middle Band')
    plt.plot(df_filtered.index, df_filtered['Real Upper Band'], 'g-', label='Upper Band')
    plt.plot(df_filtered.index, df_filtered['Real Lower Band'], 'r-', label='Lower Band')
    
    plt.title(f'Bollinger Bands ({interval_name})')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'{SYMBOL}_bbands_{interval_name}.png')
    print(f"\nBollinger Bands chart saved as {SYMBOL}_bbands_{interval_name}.png")
    plt.show()

def analyze_bandwidth(df_filtered, interval_name):
    """
    Analysis of Bollinger Bands data.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot the Band Width
    plt.plot(df_filtered.index, df_filtered['Band Width'], 'b-', label='Band Width')
    
    plt.title(f'Bollinger Bands ({interval_name})')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'{SYMBOL}_bbands_{interval_name}.png')
    print(f"\nBollinger Bands chart saved as {SYMBOL}_bbands_{interval_name}.png")
    plt.show()

def analyze_bollinger_bands(df_filtered, interval_name):
    """
    Comprehensive analysis of Bollinger Bands data.
    Combines visualization, bandwidth analysis, and anomaly detection.
    """
    if df_filtered is None or len(df_filtered) < 16:
        print(f"Insufficient data for {interval_name} analysis")
        return
    
    # Calculate band width as percentage
    df_filtered['Band Width'] = ((df_filtered['Real Upper Band'] - df_filtered['Real Lower Band']) / 
                                df_filtered['Real Middle Band']) * 100
    
    # Basic visualization
    visualize_bollinger_bands(df_filtered, interval_name)
    
    # Bandwidth analysis
    analyze_bandwidth(df_filtered, interval_name)
    
    # Anomaly detection if enough data
    if len(df_filtered) >= 20:
        detect_anomalies(df_filtered, interval_name, encoding_dim_factor=0.5, threshold_factor=3.0)
    """
    Detect anomalies in Bollinger Bands data using a PyTorch autoencoder.
    """
    # Prepare data
    features = df_filtered[['Real Upper Band', 'Real Middle Band', 'Real Lower Band', 'Band Width']]
    
    # Scale data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(features)
    
    # Split data
    train_data, test_data = train_test_split(data_scaled, test_size=0.2, random_state=42)
    
    # Define and train model
    input_shape = data_scaled.shape[1]  # Number of features
    model = define_autoencoder((input_shape,), encoding_dim_factor).to(device)
    model = train_autoencoder(model, train_data, test_data)
    
    # Get reconstruction errors
    reconstruction_errors = get_reconstruction_error(data_scaled, model)
    
    # Set threshold for anomaly detection
    threshold = np.mean(reconstruction_errors) + threshold_factor * np.std(reconstruction_errors)
    
    # Plot anomalies
    plot_anomalies(df_filtered, reconstruction_errors, threshold, interval_name)
    
    # Return anomalies for further analysis
    anomalies = reconstruction_errors > threshold
    return anomalies, reconstruction_errors, threshold

# Main execution simplified
if __name__ == "__main__":
    # Fetch data
    df_5min = fetch_and_process_bbands(params_5min, "5-minute")
    df_1hour = fetch_and_process_bbands(params_1hour, "1-hour")
    
    # Analyze each interval
    for interval_name, df in [("5-minute", df_5min), ("1-hour", df_1hour)]:
        analyze_bollinger_bands(df, interval_name)


def train_autoencoder(model, train_data, test_data, epochs=100, batch_size=32):
    """
    Train the PyTorch autoencoder model.
    """
    # Convert data to PyTorch tensors
    train_tensor = torch.FloatTensor(train_data).to(device)
    test_tensor = torch.FloatTensor(test_data).to(device)
    
    # Create DataLoader for batching
    train_dataset = TensorDataset(train_tensor, train_tensor)  # Input = Target for autoencoder
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.6f}')
    
    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_tensor)
        test_loss = criterion(test_outputs, test_tensor)
        print(f'Test Loss: {test_loss.item():.6f}')
    
    return model
