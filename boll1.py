import requests
import json
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import date, datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

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
start_date = today - timedelta(days=2)  # 48 hours ago

# Format dates for API and filtering
end_date_str = end_date.strftime('%Y-%m-%d')
start_date_str = start_date.strftime('%Y-%m-%d')
month_str = start_date.strftime('%Y-%m')

params = {
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

# Alpha Vantage API endpoint
API_URL = "https://www.alphavantage.co/query"

# --- Autoencoder Helper Functions ---
def create_sequences(data, sequence_length):
    """
    Creates sequences from the input data.
    data: pandas DataFrame with features.
    sequence_length: Number of time steps in each sequence.
    """
    xs = []
    for i in range(len(data) - sequence_length + 1):
        xs.append(data.iloc[i:(i + sequence_length)].values)
    return np.array(xs)

def define_autoencoder(input_shape, encoding_dim_factor=0.5):
    """
    Defines a simple autoencoder model.
    input_shape: tuple, shape of the input data (sequence_length, num_features) or (num_features,)
    encoding_dim_factor: determines the size of the bottleneck layer.
    """
    num_features = input_shape[-1]
    encoding_dim = int(encoding_dim_factor * num_features)
    
    if len(input_shape) == 2: # (sequence_length, num_features)
        flat_input_dim = input_shape[0] * input_shape[1]
    else: # (num_features,)
        flat_input_dim = input_shape[0]
    
    input_layer = Input(shape=(flat_input_dim,))
    encoded = Dense(int(flat_input_dim * 0.75), activation='relu')(input_layer)
    encoded = Dense(int(flat_input_dim * 0.5), activation='relu')(encoded) # Bottleneck
    decoded = Dense(int(flat_input_dim * 0.75), activation='relu')(encoded)
    decoded = Dense(flat_input_dim, activation='sigmoid')(decoded) # Sigmoid if data scaled to 0-1

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def get_reconstruction_error(data_scaled, model):
    """Calculates reconstruction error."""
    predictions = model.predict(data_scaled)
    # For flattened sequences or single timesteps, axis=1 for mse per sample
    mse = np.mean(np.power(data_scaled - predictions, 2), axis=1)
    return mse
# --- End Autoencoder Helper Functions ---

try:
    # Make the API request
    response = requests.get(API_URL, params=params)
    response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

    # Parse the JSON response
    data = response.json()

    # Check if we need to make a second API call for the current month
    # (if our 48-hour period spans two months)
    if start_date.month != end_date.month:
        second_month_str = end_date.strftime('%Y-%m')
        second_params = params.copy()
        second_params["month"] = second_month_str
        
        print(f"Fetching additional data for {second_month_str}...")
        second_response = requests.get(API_URL, params=second_params)
        second_response.raise_for_status()
        second_data = second_response.json()
        
        # Merge the two datasets
        if "Technical Analysis: BBANDS" in data and "Technical Analysis: BBANDS" in second_data:
            data["Technical Analysis: BBANDS"].update(second_data["Technical Analysis: BBANDS"])
    
    # Process the data
    if "Technical Analysis: BBANDS" in data:
        bbands_data = data["Technical Analysis: BBANDS"]
        # Convert to pandas DataFrame for easier manipulation and plotting
        df = pd.DataFrame.from_dict(bbands_data, orient='index')
        df = df.astype(float) # Convert columns to numeric
        df.index = pd.to_datetime(df.index) # Convert index to datetime
        df = df.sort_index() # Sort by date

        # Filter for the specific 48-hour period
        target_start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        target_end_date = datetime.strptime(end_date_str, '%Y-%m-%d') + timedelta(days=1) - timedelta(seconds=1)

        df_filtered = df[(df.index >= target_start_date) & (df.index <= target_end_date)]
        
        # Print summary information
        print(f"\n=== 48-Hour Bollinger Bands Analysis for {SYMBOL} ===")
        print(f"Period: {target_start_date.strftime('%Y-%m-%d %H:%M')} to {target_end_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"Interval: 5 minutes")
        print(f"Total data points: {len(df_filtered)}")
        print(f"Expected data points (perfect 48 hours): {48 * 12}")  # 12 5-min intervals per hour
        
        if len(df_filtered) < 10:
            print("\nWARNING: Very few data points available. Results may not be meaningful.")
            print("This could be due to non-trading days, market hours, or API limitations.")
        
        # Calculate band width as percentage
        df_filtered['Band Width'] = ((df_filtered['Real Upper Band'] - df_filtered['Real Lower Band']) / df_filtered['Real Middle Band']) * 100
        
        # Calculate the lowest bandwidth in a rolling window of 16 periods
        if len(df_filtered) >= 16:
            # Calculate rolling minimum of bandwidth over 16 periods
            df_filtered['Min Band Width 16'] = df_filtered['Band Width'].rolling(window=16).min()
            
            # Identify points where the current bandwidth equals the 16-period minimum
            df_filtered['Is Min Band Width'] = df_filtered['Band Width'] == df_filtered['Min Band Width 16']
            
            # Find periods with bandwidth under 6%
            narrow_bands = df_filtered[df_filtered['Band Width'] < 6]
            
            # Find consecutive periods of minimum bandwidth
            min_bandwidth_periods = df_filtered[df_filtered['Is Min Band Width'] == True]
            
            print("\n=== Bollinger Band Width Analysis ===")
            print(f"Average Band Width: {df_filtered['Band Width'].mean():.2f}%")
            print(f"Number of periods with Band Width < 6%: {len(narrow_bands)}")
            print(f"Number of minimum bandwidth points detected: {len(min_bandwidth_periods)}")
            
            if not narrow_bands.empty:
                print("\nTop 5 Narrowest Band Width Periods:")
                lowest_bw = narrow_bands.sort_values('Band Width').head(5)
                for idx, row in lowest_bw.iterrows():
                    print(f"  {idx.strftime('%Y-%m-%d %H:%M')} - Band Width: {row['Band Width']:.2f}%")
            
            # Visualize the Bollinger Bands with narrow bandwidth points highlighted
            if not df_filtered.empty:
                # Create a figure with two subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                
                # Plot Bollinger Bands on the first subplot
                ax1.plot(df_filtered.index, df_filtered['Real Middle Band'], 'b-', label='Middle Band (SMA)')
                ax1.plot(df_filtered.index, df_filtered['Real Upper Band'], 'g-', label='Upper Band')
                ax1.plot(df_filtered.index, df_filtered['Real Lower Band'], 'r-', label='Lower Band')
                ax1.fill_between(df_filtered.index, 
                                df_filtered['Real Upper Band'], 
                                df_filtered['Real Lower Band'], 
                                alpha=0.2, color='gray')
                
                # Highlight the narrow bandwidth points (< 6%)
                if not narrow_bands.empty:
                    ax1.scatter(narrow_bands.index, 
                               narrow_bands['Real Middle Band'], 
                               color='purple', s=50, zorder=5,
                               label='Band Width < 6%')
                
                # Highlight the minimum bandwidth points in the 16-period window
                if not min_bandwidth_periods.empty:
                    ax1.scatter(min_bandwidth_periods.index, 
                               min_bandwidth_periods['Real Middle Band'], 
                               color='orange', s=70, marker='*', zorder=6,
                               label='16-Period Min Band Width')
                
                ax1.set_title(f'Bollinger Bands for {SYMBOL} - 48 Hour Period (5-min intervals)')
                ax1.set_ylabel('Price')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Plot Band Width on the second subplot
                ax2.plot(df_filtered.index, df_filtered['Band Width'], 'b-', label='Band Width (%)')
                
                # Add a horizontal line at 6% threshold
                ax2.axhline(y=6, color='r', linestyle='--', label='6% Threshold')
                
                # Plot the 16-period minimum band width
                ax2.plot(df_filtered.index, df_filtered['Min Band Width 16'], 'g--', label='16-Period Min Band Width')
                
                # Highlight the narrow bandwidth points (< 6%)
                if not narrow_bands.empty:
                    ax2.scatter(narrow_bands.index, 
                               narrow_bands['Band Width'], 
                               color='purple', s=50, zorder=5,
                               label='Band Width < 6%')
                
                # Highlight the minimum bandwidth points in the 16-period window
                if not min_bandwidth_periods.empty:
                    ax2.scatter(min_bandwidth_periods.index, 
                               min_bandwidth_periods['Band Width'], 
                               color='orange', s=70, marker='*', zorder=6,
                               label='16-Period Min Band Width')
                
                ax2.set_title('Bollinger Band Width (%)')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Band Width (%)')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save the figure
                plt.savefig(f'{SYMBOL}_bollinger_bands_width_analysis.png')
                print(f"\nBollinger Bands width analysis chart saved as {SYMBOL}_bollinger_bands_width_analysis.png")
                
                # Show the figure
                plt.show()
                
                # If there are narrow bandwidth points, create a zoomed-in view of the narrowest one
                if len(narrow_bands) > 0:
                    # Get the narrowest bandwidth point
                    narrowest_point = narrow_bands.sort_values('Band Width').iloc[0]
                    narrowest_time = narrowest_point.name
                    
                    # Create a window around this point (8 periods before and after)
                    window_start = max(0, df_filtered.index.get_loc(narrowest_time) - 8)
                    window_end = min(len(df_filtered), df_filtered.index.get_loc(narrowest_time) + 9)
                    window_df = df_filtered.iloc[window_start:window_end]
                    
                    # Plot the zoomed view
                    plt.figure(figsize=(10, 6))
                    plt.plot(window_df.index, window_df['Real Middle Band'], 'b-', label='Middle Band (SMA)')
                    plt.plot(window_df.index, window_df['Real Upper Band'], 'g-', label='Upper Band')
                    plt.plot(window_df.index, window_df['Real Lower Band'], 'r-', label='Lower Band')
                    plt.fill_between(window_df.index, 
                                    window_df['Real Upper Band'], 
                                    window_df['Real Lower Band'], 
                                    alpha=0.2, color='gray')
                    
                    # Highlight the narrowest bandwidth point
                    plt.scatter([narrowest_time], 
                               [narrowest_point['Real Middle Band']], 
                               color='purple', s=100, zorder=5,
                               label=f'Narrowest Band Width: {narrowest_point["Band Width"]:.2f}%')
                    
                    plt.title(f'Zoomed View of Narrowest Band Width Point - {narrowest_time.strftime("%Y-%m-%d %H:%M")}')
                    plt.xlabel('Time')
                    plt.ylabel('Price')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Save the zoomed figure
                    plt.savefig(f'{SYMBOL}_narrowest_bandwidth_zoomed.png')
                    print(f"\nZoomed view of narrowest bandwidth point saved as {SYMBOL}_narrowest_bandwidth_zoomed.png")
                    plt.show()
        else:
            print("\nInsufficient data for 16-period rolling window analysis. Need at least 16 data points.")
            
        # Continue with the rest of your analysis...
        # Visualize the Bollinger Bands
        if not df_filtered.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(df_filtered.index, df_filtered['Real Middle Band'], 'b-', label='Middle Band (SMA)')
            plt.plot(df_filtered.index, df_filtered['Real Upper Band'], 'g-', label='Upper Band')
            plt.plot(df_filtered.index, df_filtered['Real Lower Band'], 'r-', label='Lower Band')
            plt.fill_between(df_filtered.index, 
                            df_filtered['Real Upper Band'], 
                            df_filtered['Real Lower Band'], 
                            alpha=0.2, color='gray')
            
            plt.title(f'Bollinger Bands for {SYMBOL} - 48 Hour Period (5-min intervals)')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(f'{SYMBOL}_bollinger_bands_48h.png')
            print(f"\nBollinger Bands chart saved as {SYMBOL}_bollinger_bands_48h.png")
            
            # Show the figure
            plt.show()
            
            # Calculate band width and volatility metrics
            df_filtered['Band Width'] = (df_filtered['Real Upper Band'] - df_filtered['Real Lower Band']) / df_filtered['Real Middle Band']
            
            # Find periods of high volatility (wide bands)
            volatility_threshold = df_filtered['Band Width'].mean() + df_filtered['Band Width'].std()
            high_volatility = df_filtered[df_filtered['Band Width'] > volatility_threshold]
            
            print("\n=== Volatility Analysis ===")
            print(f"Average Band Width: {df_filtered['Band Width'].mean():.4f}")
            print(f"High Volatility Threshold: {volatility_threshold:.4f}")
            print(f"High Volatility Periods: {len(high_volatility)} intervals")
            
            if not high_volatility.empty:
                print("\nTop 5 Highest Volatility Periods:")
                top_volatility = high_volatility.sort_values('Band Width', ascending=False).head(5)
                for idx, row in top_volatility.iterrows():
                    print(f"  {idx.strftime('%Y-%m-%d %H:%M')} - Band Width: {row['Band Width']:.4f}")
            
            # Run autoencoder analysis if sufficient data
            if len(df_filtered) >= 20:  # Minimum threshold for meaningful analysis
                # --- Autoencoder Analysis ---
                features = ['Real Lower Band', 'Real Middle Band', 'Real Upper Band']
                data_for_ae = df_filtered[features].copy()

                # Scale data
                scaler = MinMaxScaler()
                data_scaled = scaler.fit_transform(data_for_ae)
                
                # Create sequences for anomaly detection
                sequence_length = 6  # 30 minutes (6 x 5-min intervals)
                
                print(f"\n=== Anomaly Detection (30-minute sequences) ===")
                sequences = create_sequences(pd.DataFrame(data_scaled, columns=features), sequence_length)
                
                if sequences.shape[0] > 10:  # Ensure enough sequences for training
                    num_sequences = sequences.shape[0]
                    num_features_per_step = sequences.shape[2]
                    sequences_flattened = sequences.reshape(num_sequences, sequence_length * num_features_per_step)
                    
                    # Split data
                    train_size = int(len(sequences_flattened) * 0.7)
                    train_data = sequences_flattened[:train_size]
                    test_data = sequences_flattened[train_size:]
                    
                    if len(train_data) > 5 and len(test_data) > 0:
                        # Define and train autoencoder
                        input_shape = (sequence_length * num_features_per_step,)
                        autoencoder = define_autoencoder(input_shape)
                        
                        print("Training autoencoder for anomaly detection...")
                        autoencoder.fit(train_data, train_data,
                                        epochs=20, batch_size=16, shuffle=True, verbose=0,
                                        validation_data=(test_data, test_data),
                                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')])
                        
                        # Detect anomalies
                        reconstruction_error = get_reconstruction_error(test_data, autoencoder)
                        threshold = np.mean(reconstruction_error) + 2 * np.std(reconstruction_error)
                        anomalies_indices = np.where(reconstruction_error > threshold)[0]
                        
                        print(f"Found {len(anomalies_indices)} anomalous 30-minute sequences (threshold: {threshold:.4f})")
                        
                        if len(anomalies_indices) > 0:
                            print("\nTop 5 Anomalies:")
                            # Sort anomalies by reconstruction error
                            sorted_indices = anomalies_indices[np.argsort(reconstruction_error[anomalies_indices])[::-1]]
                            top_anomalies = sorted_indices[:min(5, len(sorted_indices))]
                            
                            for i, anom_idx in enumerate(top_anomalies):
                                original_df_idx = train_size + anom_idx
                                if original_df_idx < len(df_filtered) - sequence_length + 1:
                                    timestamp = df_filtered.index[original_df_idx]
                                    error = reconstruction_error[anom_idx]
                                    print(f"  {i+1}. {timestamp.strftime('%Y-%m-%d %H:%M')} - Error: {error:.4f}")
                            
                            # Visualize the top anomaly
                            if len(top_anomalies) > 0:
                                top_anom_idx = train_size + top_anomalies[0]
                                if top_anom_idx < len(df_filtered) - sequence_length + 1:
                                    anom_start = df_filtered.index[top_anom_idx]
                                    anom_end = df_filtered.index[min(top_anom_idx + sequence_length - 1, len(df_filtered) - 1)]
                                    
                                    # Plot the anomalous sequence
                                    plt.figure(figsize=(10, 5))
                                    anom_df = df_filtered.loc[anom_start:anom_end]
                                    
                                    plt.plot(anom_df.index, anom_df['Real Middle Band'], 'b-', label='Middle Band')
                                    plt.plot(anom_df.index, anom_df['Real Upper Band'], 'g-', label='Upper Band')
                                    plt.plot(anom_df.index, anom_df['Real Lower Band'], 'r-', label='Lower Band')
                                    plt.fill_between(anom_df.index, 
                                                    anom_df['Real Upper Band'], 
                                                    anom_df['Real Lower Band'], 
                                                    alpha=0.2, color='gray')
                                    
                                    plt.title(f'Top Anomalous Sequence - {anom_start.strftime("%Y-%m-%d %H:%M")}')
                                    plt.xlabel('Time')
                                    plt.ylabel('Price')
                                    plt.grid(True, alpha=0.3)
                                    plt.legend()
                                    plt.xticks(rotation=45)
                                    plt.tight_layout()
                                    
                                    # Save the anomaly figure
                                    plt.savefig(f'{SYMBOL}_top_anomaly.png')
                                    print(f"\nTop anomaly chart saved as {SYMBOL}_top_anomaly.png")
                                    plt.show()
                    else:
                        print("Not enough data for training/testing the autoencoder after splitting.")
                else:
                    print("Not enough sequences for meaningful anomaly detection.")
            else:
                print("\nInsufficient data for autoencoder analysis. Need at least 20 data points.")
        else:
            print("\nNo data found for the specified 48-hour period.")
    else:
        print("\nNo Bollinger Bands data found in the API response.")
        print("API Response:", data)
        
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()