import requests
import json
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import date, datetime, timedelta # Modified: Added datetime, timedelta
import numpy as np # New import
from sklearn.preprocessing import MinMaxScaler # New import
from tensorflow.keras.models import Model # New import
from tensorflow.keras.layers import Input, Dense, LSTM # New import
from tensorflow.keras.callbacks import EarlyStopping # New import

load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
API_KEY = ALPHA_VANTAGE_API_KEY
SYMBOL = 'AAPL' # Apple Inc.

# API Parameters
# Modified to target a specific month for historical intraday data
# For May 19th/20th, let's target May 2024.
# Note: May 19, 2024 was a Sunday, May 20, 2024 was a Monday.
# Adjust "month" if you need a different year/month.
params = {
    "function": "BBANDS",
    "symbol": SYMBOL,
    "interval": "5min",        # Set to "5min" for 5-min details, or "60min" for 1-hr details
    "month": "2024-05",
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

    # Print the raw JSON data (or process it further)
    # print(json.dumps(data, indent=4)) # Optionally keep this for debugging

    # --- Optional: Further processing and visualization ---
    # You can add more code here to process the data, for example, using pandas
    # and matplotlib for plotting if the data is successfully retrieved.

    # Example (if 'Technical Analysis: BBANDS' is in the response):
    if "Technical Analysis: BBANDS" in data:
        bbands_data = data["Technical Analysis: BBANDS"]
        # Convert to pandas DataFrame for easier manipulation and plotting
        # import pandas as pd # Already imported at the top
        # from datetime import date # Already imported at the top
        df = pd.DataFrame.from_dict(bbands_data, orient='index')
        df = df.astype(float) # Convert columns to numeric
        df.index = pd.to_datetime(df.index) # Convert index to datetime
        df = df.sort_index() # Sort by date

        print("\nFull Processed DataFrame (first 5 rows from fetched month):")
        print(df.head())

        # Filter for the specific dates: May 19, 2024 and May 20, 2024
        # Note: May 19, 2024 was a Sunday. May 20, 2024 was a Monday.
        # You might only get data for May 20th.
        target_start_date = datetime(2024, 5, 19)
        target_end_date = datetime(2024, 5, 20, 23, 59, 59) # Include whole of May 20th

        if not df.empty:
            df_filtered = df[(df.index >= target_start_date) & (df.index <= target_end_date)]
        else:
            df_filtered = pd.DataFrame()

        print(f"\nData for {target_start_date.strftime('%Y-%m-%d')} and {target_end_date.strftime('%Y-%m-%d')}:")
        if not df_filtered.empty:
            print(df_filtered)
        else:
            print("No data found for the specified dates in the fetched data.")
            print("This could be due to non-trading days or API data limitations for the specified month/year.")


        if not df_filtered.empty:
            print(f"\nFiltered DataFrame for Autoencoder analysis (first 5 rows):")
            print(df_filtered.head())
            print(f"Number of data points for autoencoders: {len(df_filtered)}")
            if len(df_filtered) < 10: # Arbitrary small number
                print("WARNING: Very few data points for autoencoder training and testing.")
                print("Results will likely not be meaningful. Autoencoders need more data to learn patterns.")


            # --- Start Autoencoder Logic ---
            features = ['Real Lower Band', 'Real Middle Band', 'Real Upper Band']
            data_for_ae = df_filtered[features].copy()

            # Scale data
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data_for_ae)
            
            # --- 1-Hour Period Autoencoder ---
            print("\n--- 1-Hour Period Autoencoder Analysis ---")
            input_shape_1hr = (data_scaled.shape[1],) # (num_features,)
            
            train_size_1hr = int(len(data_scaled) * 0.7)
            train_data_1hr = data_scaled[:train_size_1hr]
            test_data_1hr = data_scaled[train_size_1hr:]
            
            if len(train_data_1hr) > 2 and len(test_data_1hr) > 0: # Reduced minimum for very small datasets
                autoencoder_1hr = define_autoencoder(input_shape_1hr)
                print("Training 1-Hour Autoencoder (conceptual)...")
                autoencoder_1hr.fit(train_data_1hr, train_data_1hr,
                                    epochs=20, batch_size=16, shuffle=True, verbose=0,
                                    validation_data=(test_data_1hr, test_data_1hr),
                                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')])

                reconstruction_error_1hr = get_reconstruction_error(test_data_1hr, autoencoder_1hr)
                threshold_1hr = np.mean(reconstruction_error_1hr) + 2 * np.std(reconstruction_error_1hr)
                anomalies_1hr_indices_in_test = np.where(reconstruction_error_1hr > threshold_1hr)[0] # Renamed for clarity
                
                print(f"1-Hour AE: Found {len(anomalies_1hr_indices_in_test)} anomalies (threshold: {threshold_1hr:.4f})")
                
                if len(anomalies_1hr_indices_in_test) > 0:
                    print("--- Detailed 1-Hour Anomalies ---")
                    # original_indices_1hr = df_filtered.index[train_size_1hr:][anomalies_1hr_indices_in_test] # Original way to get timestamps
                    for i, anom_idx_in_test in enumerate(anomalies_1hr_indices_in_test):
                        # Get the actual index in the original df_filtered
                        original_df_idx = train_size_1hr + anom_idx_in_test
                        timestamp = df_filtered.index[original_df_idx]
                        anomaly_error = reconstruction_error_1hr[anom_idx_in_test]
                        feature_values = df_filtered.iloc[original_df_idx][features]
                        
                        print(f"  Anomaly {i+1}:")
                        print(f"    Timestamp: {timestamp}")
                        print(f"    Reconstruction Error: {anomaly_error:.4f}")
                        print(f"    Feature Values:\n{feature_values.to_string()}")
                        print("-" * 20)
            else:
                print("Not enough data for 1-Hour Autoencoder training/testing example after splitting.")
                print(f"  Available data points: {len(data_scaled)}, Train size: {len(train_data_1hr)}, Test size: {len(test_data_1hr)}")

            # --- Multi-Interval Sequence Autoencoder (Previously "4-Hour") ---
            # This section creates sequences from the base interval data.
            # The sequence_length determines how many base intervals form one sequence.
            base_interval_str = params.get("interval", "unknown_interval")
            sequence_length_multi = 4 # Number of base intervals per sequence
            
            # Calculate actual duration of the sequence
            interval_value = int(base_interval_str.replace('min','')) if 'min' in base_interval_str else 0
            if interval_value > 0:
                sequence_duration_minutes = sequence_length_multi * interval_value
                if sequence_duration_minutes >= 60:
                    sequence_duration_str = f"{sequence_duration_minutes // 60} hour(s)"
                    if sequence_duration_minutes % 60 > 0:
                        sequence_duration_str += f" {sequence_duration_minutes % 60} min"
                else:
                    sequence_duration_str = f"{sequence_duration_minutes} min"
            else:
                sequence_duration_str = f"{sequence_length_multi} base intervals (unknown duration)"

            print(f"\n--- Multi-Interval Sequence Autoencoder Analysis ({sequence_duration_str} sequences) ---")
            print(f"Base interval: {base_interval_str}, Sequence length: {sequence_length_multi} intervals.")
            
            sequences_multi_original_shape = create_sequences(pd.DataFrame(data_scaled, columns=features), sequence_length_multi)
            
            if sequences_multi_original_shape.shape[0] > 0:
                num_sequences = sequences_multi_original_shape.shape[0]
                num_features_per_step = sequences_multi_original_shape.shape[2]
                sequences_multi_flattened = sequences_multi_original_shape.reshape(num_sequences, sequence_length_multi * num_features_per_step)

                input_shape_multi = (sequence_length_multi * num_features_per_step,)

                train_size_multi = int(len(sequences_multi_flattened) * 0.7)
                train_data_multi = sequences_multi_flattened[:train_size_multi]
                test_data_multi = sequences_multi_flattened[train_size_multi:]

                if len(train_data_multi) > 2 and len(test_data_multi) > 0: # Reduced minimum
                    autoencoder_multi = define_autoencoder(input_shape_multi)
                    print(f"Training Multi-Interval ({sequence_duration_str}) Autoencoder...")
                    autoencoder_multi.fit(train_data_multi, train_data_multi,
                                        epochs=20, batch_size=16, shuffle=True, verbose=0,
                                        validation_data=(test_data_multi, test_data_multi),
                                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')])
                    
                    reconstruction_error_multi = get_reconstruction_error(test_data_multi, autoencoder_multi)
                    threshold_multi = np.mean(reconstruction_error_multi) + 2 * np.std(reconstruction_error_multi)
                    anomalies_multi_indices_in_test = np.where(reconstruction_error_multi > threshold_multi)[0]

                    print(f"Multi-Interval ({sequence_duration_str}) AE: Found {len(anomalies_multi_indices_in_test)} sequence anomalies (threshold: {threshold_multi:.4f})")
                    
                    if len(anomalies_multi_indices_in_test) > 0:
                        print(f"--- Detailed Multi-Interval ({sequence_duration_str}) Sequence Anomalies ---")
                        for i, anom_idx_in_test in enumerate(anomalies_multi_indices_in_test):
                            # original_df_idx is the start of the anomalous sequence in df_filtered
                            original_df_idx_start_of_sequence = train_size_multi + anom_idx_in_test
                            
                            # Ensure the sequence start index is valid for df_filtered
                            if original_df_idx_start_of_sequence < (len(df_filtered) - sequence_length_multi + 1):
                                timestamp_start = df_filtered.index[original_df_idx_start_of_sequence]
                                anomaly_error = reconstruction_error_multi[anom_idx_in_test]
                                
                                # Get feature values for the entire anomalous sequence
                                sequence_feature_values_list = []
                                for step in range(sequence_length_multi):
                                    step_idx = original_df_idx_start_of_sequence + step
                                    if step_idx < len(df_filtered):
                                        sequence_feature_values_list.append(df_filtered.iloc[step_idx][features].to_string())
                                    else:
                                        sequence_feature_values_list.append("Data point out of bounds")
                                sequence_feature_values_str = "\n---\n".join(sequence_feature_values_list)

                                print(f"  Anomaly Sequence {i+1}:")
                                print(f"    Start Timestamp: {timestamp_start}")
                                print(f"    Reconstruction Error (for sequence): {anomaly_error:.4f}")
                                print(f"    Feature Values for each step in sequence:\n{sequence_feature_values_str}")
                                print("-" * 20)
                            else:
                                print(f"  Anomaly Sequence {i+1} (index {original_df_idx_start_of_sequence}) out of bounds for detailed view.")
                else:
                    print(f"Not enough data for Multi-Interval ({sequence_duration_str}) Autoencoder training/testing example after splitting.")
                    print(f"  Available sequences: {len(sequences_multi_flattened)}, Train sequences: {len(train_data_multi)}, Test sequences: {len(test_data_multi)}")
            else:
                print(f"Not enough data to create {sequence_duration_str} sequences from the filtered data.")
            # --- 5-Minute Period Autoencoder (More Sensitive) ---
            print("\n--- 4-Hour Period Autoencoder Analysis ---")
            sequence_length_4hr = 4 
            
            sequences_4hr_original_shape = create_sequences(pd.DataFrame(data_scaled, columns=features), sequence_length_4hr)
            
            if sequences_4hr_original_shape.shape[0] > 0:
                num_sequences = sequences_4hr_original_shape.shape[0]
                num_features_per_step = sequences_4hr_original_shape.shape[2]
                sequences_4hr_flattened = sequences_4hr_original_shape.reshape(num_sequences, sequence_length_4hr * num_features_per_step)

                input_shape_4hr = (sequence_length_4hr * num_features_per_step,)

                train_size_4hr = int(len(sequences_4hr_flattened) * 0.7)
                train_data_4hr = sequences_4hr_flattened[:train_size_4hr]
                test_data_4hr = sequences_4hr_flattened[train_size_4hr:]

                if len(train_data_4hr) > 2 and len(test_data_4hr) > 0: # Reduced minimum
                    autoencoder_4hr = define_autoencoder(input_shape_4hr)
                    print("Training 4-Hour Autoencoder (conceptual)...")
                    autoencoder_4hr.fit(train_data_4hr, train_data_4hr,
                                        epochs=20, batch_size=16, shuffle=True, verbose=0,
                                        validation_data=(test_data_4hr, test_data_4hr),
                                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')])
                    
                    reconstruction_error_4hr = get_reconstruction_error(test_data_4hr, autoencoder_4hr)
                    threshold_4hr = np.mean(reconstruction_error_4hr) + 2 * np.std(reconstruction_error_4hr)
                    anomalies_4hr_indices = np.where(reconstruction_error_4hr > threshold_4hr)[0]

                    print(f"4-Hour AE: Found {len(anomalies_4hr_indices)} sequence anomalies (threshold: {threshold_4hr:.4f})")
                    if anomalies_4hr_indices.size > 0:
                        print("Anomaly start timestamps (4-hour sequences):")
                        for anom_idx in anomalies_4hr_indices:
                            original_df_idx = train_size_4hr + anom_idx
                            if original_df_idx < len(df_filtered) - sequence_length_4hr + 1:
                                print(df_filtered.index[original_df_idx])
                else:
                    print("Not enough data for 4-Hour Autoencoder training/testing example after splitting.")
                    print(f"  Available sequences: {len(sequences_4hr_flattened)}, Train sequences: {len(train_data_4hr)}, Test sequences: {len(test_data_4hr)}")
            else:
                print("Not enough data to create 4-hour sequences from the filtered data.")
            # --- End Autoencoder Logic ---

            # --- 5-Minute Period Autoencoder (More Sensitive) ---
            print("\n--- 5-Minute Period Autoencoder Analysis ---")
            sequence_length_5min = 4 
            
            sequences_5min_original_shape = create_sequences(pd.DataFrame(data_scaled, columns=features), sequence_length_5min)
            
            if sequences_5min_original_shape.shape[0] > 0:
                num_sequences = sequences_5min_original_shape.shape[0]
                num_features_per_step = sequences_5min_original_shape.shape[2]
                sequences_5min_flattened = sequences_5min_original_shape.reshape(num_sequences, sequence_length_5min * num_features_per_step)

                input_shape_5min = (sequence_length_5min * num_features_per_step,)

                train_size_5min = int(len(sequences_5min_flattened) * 0.7)
                train_data_5min = sequences_5min_flattened[:train_size_5min]
                test_data_5min = sequences_5min_flattened[train_size_5min:]

                if len(train_data_5min) > 2 and len(test_data_5min) > 0: # Reduced minimum
                    autoencoder_5min = define_autoencoder(input_shape_5min)
                    print("Training 5-Minute Autoencoder...")
                    autoencoder_5min.fit(train_data_5min, train_data_5min,
                                        epochs=20, batch_size=16, shuffle=True, verbose=0,
                                        validation_data=(test_data_5min, test_data_5min),
                                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')])
                    
                    reconstruction_error_5min = get_reconstruction_error(test_data_5min, autoencoder_5min)
                    threshold_5min = np.mean(reconstruction_error_5min) + 2 * np.std(reconstruction_error_5min)
                    anomalies_5min_indices = np.where(reconstruction_error_5min > threshold_5min)[0]

                    print(f"5-Minute AE: Found {len(anomalies_5min_indices)} sequence anomalies (threshold: {threshold_5min:.4f})")
                    if anomalies_5min_indices.size > 0:
                        print("Anomaly start timestamps (5-minute sequences):")
                        for anom_idx in anomalies_5min_indices:
                            original_df_idx = train_size_5min + anom_idx
                            if original_df_idx < len(df_filtered) - sequence_length_5min + 1:
                                print(df_filtered.index[original_df_idx])
                else:
                    print("Not enough data for 5-Minute Autoencoder training/testing example after splitting.")
                    print(f"  Available sequences: {len(sequences_5min_flattened)}, Train sequences: {len(train_data_5min)}, Test sequences: {len(test_data_5min)}")
            else:
                print("Not enough data to create 5-minute sequences from the filtered data.")
            # --- End Autoencoder Logic ---

            # After your autoencoder analysis, add this plotting code:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.ticker import MaxNLocator

            # Filter for just trading hours (9:30 AM - 4:00 PM ET)
            # For May 20, 2024 (Monday)
            trading_start = datetime(2024, 5, 20, 9, 30)  # 9:30 AM ET
            trading_end = datetime(2024, 5, 20, 16, 0)    # 4:00 PM ET

            # Filter data for trading hours
            trading_hours_data = df_filtered[(df_filtered.index >= trading_start) & 
                                             (df_filtered.index <= trading_end)]

            if not trading_hours_data.empty:
                # Create Figure 1 - Bollinger Bands with Anomalies
                plt.figure(figsize=(14, 8))
                plt.title('AAPL Bollinger Bands with Anomaly Detection (May 20, 2024 Trading Hours)', fontsize=14)
                
                # Plot Bollinger Bands
                plt.plot(trading_hours_data.index, trading_hours_data['Real Middle Band'], 'b-', label='Middle Band (SMA)')
                plt.plot(trading_hours_data.index, trading_hours_data['Real Upper Band'], 'g-', label='Upper Band')
                plt.plot(trading_hours_data.index, trading_hours_data['Real Lower Band'], 'r-', label='Lower Band')
                
                # Plot anomalies if they exist
                # 1-Hour anomalies
                if 'anomalies_1hr_indices_in_test' in locals() and len(anomalies_1hr_indices_in_test) > 0:
                    anomaly_timestamps = []
                    for anom_idx in anomalies_1hr_indices_in_test:
                        original_df_idx = train_size_1hr + anom_idx
                        if original_df_idx < len(df_filtered):
                            timestamp = df_filtered.index[original_df_idx]
                            if trading_start <= timestamp <= trading_end:
                                anomaly_timestamps.append(timestamp)
                        
                        if anomaly_timestamps:
                            plt.scatter(anomaly_timestamps, 
                                       trading_hours_data.loc[anomaly_timestamps]['Real Middle Band'],
                                       color='purple', marker='o', s=100, label='1-Hr Anomaly')
                    
                    # 5-Minute anomalies
                    if 'anomalies_5min_indices' in locals() and anomalies_5min_indices.size > 0:
                        anomaly_timestamps_5min = []
                        for anom_idx in anomalies_5min_indices:
                            original_df_idx = train_size_5min + anom_idx
                            if original_df_idx < len(df_filtered) - sequence_length_5min + 1:
                                timestamp = df_filtered.index[original_df_idx]
                                if trading_start <= timestamp <= trading_end:
                                    anomaly_timestamps_5min.append(timestamp)
                        
                        if anomaly_timestamps_5min:
                            plt.scatter(anomaly_timestamps_5min,
                                       trading_hours_data.loc[anomaly_timestamps_5min]['Real Middle Band'],
                                       color='red', marker='x', s=80, label='5-Min Anomaly')
                    
                    # Multi-interval anomalies
                    if 'anomalies_multi_indices_in_test' in locals() and len(anomalies_multi_indices_in_test) > 0:
                        anomaly_timestamps_multi = []
                        for anom_idx in anomalies_multi_indices_in_test:
                            original_df_idx = train_size_multi + anom_idx
                            if original_df_idx < len(df_filtered) - sequence_length_multi + 1:
                                timestamp = df_filtered.index[original_df_idx]
                                if trading_start <= timestamp <= trading_end:
                                    anomaly_timestamps_multi.append(timestamp)
                        
                        if anomaly_timestamps_multi:
                            plt.scatter(anomaly_timestamps_multi,
                                       trading_hours_data.loc[anomaly_timestamps_multi]['Real Middle Band'],
                                       color='orange', marker='s', s=80, label=f'Multi-Interval ({sequence_duration_str}) Anomaly')
                    
                    # Format x-axis to show all trading hour intervals
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
                    plt.gca().xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
                    
                    plt.xlabel('Time (May 20, 2024)', fontsize=12)
                    plt.ylabel('Price', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.legend(loc='best')
                    
                    # Rotate date labels for better readability
                    plt.gcf().autofmt_xdate()
                    
                    # Add a note about trading hours
                    plt.figtext(0.5, 0.01, 'Note: Normal Trading Hours (9:30 AM - 4:00 PM ET)', 
                                ha='center', fontsize=10, style='italic')
                    
                    plt.tight_layout()
                    plt.savefig('aapl_bollinger_bands_with_anomalies.png', dpi=300)
                    plt.show()
            else:
                print("No data available for trading hours on May 20, 2024.")
            # Example plotting (requires matplotlib)
            import matplotlib.pyplot as plt # Ensure matplotlib is imported
            plt.figure(figsize=(15,7)) # Adjusted figure size
            plt.plot(df_filtered.index, df_filtered['Real Middle Band'], label='Middle Band')
            plt.plot(df_filtered.index, df_filtered['Real Upper Band'], label='Upper Band', color='red')
            plt.plot(df_filtered.index, df_filtered['Real Lower Band'], label='Lower Band', color='green')
            plt.fill_between(df_filtered.index, df_filtered['Real Lower Band'], df_filtered['Real Upper Band'], color='gray', alpha=0.3)
            
            # --- Plotting Anomalies ---
            if 'original_indices_1hr' in locals() and not original_indices_1hr.empty:
                 plt.scatter(original_indices_1hr, 
                            df_filtered.loc[original_indices_1hr]['Real Middle Band'], # Plot on middle band
                            color='purple', marker='o', s=100, label='1-Hr Anomaly')

            if 'anomalies_5min_indices' in locals() and anomalies_5min_indices.size > 0 and 'train_size_5min' in locals() and 'sequence_length_5min' in locals():
                anom_starts_5min_plot = []
                # For plotting, we need the indices from the 'multi-interval' section
                # Assuming anomalies_multi_indices_in_test and train_size_multi are now the correct variables
                if 'anomalies_multi_indices_in_test' in locals() and len(anomalies_multi_indices_in_test) > 0:
                    for anom_idx in anomalies_multi_indices_in_test:
                        original_df_idx = train_size_multi + anom_idx # This is index in sequences_multi_flattened
                        # The corresponding index in df_filtered is the start of the sequence
                        if original_df_idx < (len(df_filtered) - sequence_length_multi + 1):
                             anom_starts_5min_plot.append(df_filtered.index[original_df_idx])
                    if anom_starts_5min_plot:
                        plt.scatter(anom_starts_5min_plot,
                                    df_filtered.loc[anom_starts_5min_plot]['Real Middle Band'], # Plot on middle band
                                    color='orange', marker='X', s=150, label=f'{sequence_duration_str} Anomaly Start')
            
            if 'original_indices_5min' in locals() and not original_indices_5min.empty:
                 plt.scatter(original_indices_5min, 
                            df_filtered.loc[original_indices_5min]['Real Middle Band'], # Plot on middle band
                            color='cyan', marker='P', s=120, label='5-Min Anomaly (Sensitive)')
            # --- End Plotting Anomalies ---

            plt.title(f'Bollinger Bands for {SYMBOL} (5min interval)') # Updated title
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            # print(f"\nNo data found for the period {start_date} to {end_date}.") # Original message
            print(f"\nNo data found for the period for analysis.") # Updated message

    else:
        print("\nCould not find 'Technical Analysis: BBANDS' in the API response.")
        # --- Additions for more API error info ---
        if "Error Message" in data:
            print(f"API Error: {data['Error Message']}")
        elif "Information" in data: 
            print(f"API Info: {data['Information']}") # Often indicates API call limit reached or other issues
        # --- End Additions ---

except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
except requests.exceptions.ConnectionError as conn_err:
    print(f"Connection error occurred: {conn_err}")
except requests.exceptions.Timeout as timeout_err:
    print(f"Timeout error occurred: {timeout_err}")
except requests.exceptions.RequestException as req_err:
    print(f"An error occurred during the request: {req_err}")
except json.JSONDecodeError:
    print("Failed to decode JSON response. Raw response:")
    print(response.text)