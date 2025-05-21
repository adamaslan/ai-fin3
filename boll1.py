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
params = {
    "function": "BBANDS",
    "symbol": SYMBOL,
    "interval": "60min",       # Changed to 60min
    "time_period": "20",       # BBands calculation period (e.g., 20 hours for 60min interval)
    "series_type": "close",    # open, high, low, close
    "apikey": API_KEY,
    "datatype": "json"         # json or csv
    # Optional parameters (can be added if needed):
    # "month": "2009-01",      # For intraday intervals, YYYY-MM format
    # "nbdevup": "2",          # Standard deviation multiplier for upper band
    # "nbdevdn": "2",          # Standard deviation multiplier for lower band
    # "matype": "0"            # Moving average type (0 for SMA, 1 for EMA, etc.)
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

        print("\nFull Processed DataFrame (first 5 rows):")
        print(df.head())

        # Filter for the date range - adjust for intraday data
        # start_date = '2023-01-01' # Original daily filter
        # end_date = date.today().strftime('%Y-%m-%d') # Original daily filter
        # df_filtered = df[(df.index >= start_date) & (df.index <= end_date)] # Original daily filter
        
        # For intraday, filter for a more recent period, e.g., last 30 days
        if not df.empty:
            df_filtered = df.last('30D') # Example: last 30 days of 60min data
            if df_filtered.empty: # Fallback if 'last 30D' results in empty
                df_filtered = df 
        else:
            df_filtered = pd.DataFrame()


        if not df_filtered.empty:
            print(f"\nFiltered DataFrame for analysis (first 5 rows):")
            print(df_filtered.head())

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
            
            if len(train_data_1hr) > 5 and len(test_data_1hr) > 0:
                autoencoder_1hr = define_autoencoder(input_shape_1hr)
                print("Training 1-Hour Autoencoder (conceptual)...")
                autoencoder_1hr.fit(train_data_1hr, train_data_1hr,
                                    epochs=20, batch_size=16, shuffle=True, verbose=0,
                                    validation_data=(test_data_1hr, test_data_1hr),
                                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')])

                reconstruction_error_1hr = get_reconstruction_error(test_data_1hr, autoencoder_1hr)
                threshold_1hr = np.mean(reconstruction_error_1hr) + 2 * np.std(reconstruction_error_1hr)
                anomalies_1hr_indices = np.where(reconstruction_error_1hr > threshold_1hr)[0]
                
                print(f"1-Hour AE: Found {len(anomalies_1hr_indices)} anomalies (threshold: {threshold_1hr:.4f})")
                original_indices_1hr = df_filtered.index[train_size_1hr:][anomalies_1hr_indices]
                if not original_indices_1hr.empty:
                    print("Anomaly timestamps (1-hour):")
                    for ts in original_indices_1hr:
                        print(ts)
            else:
                print("Not enough data for 1-Hour Autoencoder training/testing example.")

            # --- 4-Hour Period Autoencoder ---
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

                if len(train_data_4hr) > 5 and len(test_data_4hr) > 0:
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
                    print("Not enough data for 4-Hour Autoencoder training/testing example.")
            else:
                print("Not enough data to create 4-hour sequences.")
            # --- End Autoencoder Logic ---

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

            if 'anomalies_4hr_indices' in locals() and anomalies_4hr_indices.size > 0 and 'train_size_4hr' in locals() and 'sequence_length_4hr' in locals():
                anom_starts_4hr_plot = []
                for anom_idx in anomalies_4hr_indices:
                    original_df_idx = train_size_4hr + anom_idx # This is index in sequences_4hr_flattened
                    # The corresponding index in df_filtered is the start of the sequence
                    if original_df_idx < (len(df_filtered) - sequence_length_4hr + 1):
                         anom_starts_4hr_plot.append(df_filtered.index[original_df_idx])
                if anom_starts_4hr_plot:
                    plt.scatter(anom_starts_4hr_plot,
                                df_filtered.loc[anom_starts_4hr_plot]['Real Middle Band'], # Plot on middle band
                                color='orange', marker='X', s=150, label='4-Hr Anomaly Start')
            # --- End Plotting Anomalies ---

            plt.title(f'Bollinger Bands for {SYMBOL} (60min interval)') # Updated title
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