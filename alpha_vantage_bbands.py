import requests
import json

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
API_KEY = 'YOUR_API_KEY'
SYMBOL = 'AAPL' # Apple Inc.

# API Parameters
params = {
    "function": "BBANDS",
    "symbol": SYMBOL,
    "interval": "daily",       # 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
    "time_period": "20",       # Number of data points (e.g., 20 days)
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

try:
    # Make the API request
    response = requests.get(API_URL, params=params)
    response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

    # Parse the JSON response
    data = response.json()

    # Print the raw JSON data (or process it further)
    print(json.dumps(data, indent=4))

    # --- Optional: Further processing and visualization ---
    # You can add more code here to process the data, for example, using pandas
    # and matplotlib for plotting if the data is successfully retrieved.

    # Example (if 'Technical Analysis: BBANDS' is in the response):
    # if "Technical Analysis: BBANDS" in data:
    #     bbands_data = data["Technical Analysis: BBANDS"]
    #     # Convert to pandas DataFrame for easier manipulation and plotting
    #     import pandas as pd
    #     df = pd.DataFrame.from_dict(bbands_data, orient='index')
    #     df = df.astype(float) # Convert columns to numeric
    #     df.index = pd.to_datetime(df.index) # Convert index to datetime
    #     df = df.sort_index() # Sort by date
    #
    #     print("\nProcessed DataFrame:")
    #     print(df.head())
    #
    #     # Example plotting (requires matplotlib)
    #     # import matplotlib.pyplot as plt
    #     # plt.figure(figsize=(12,6))
    #     # plt.plot(df.index, df['Real Middle Band'], label='Middle Band')
    #     # plt.plot(df.index, df['Real Upper Band'], label='Upper Band', color='red')
    #     # plt.plot(df.index, df['Real Lower Band'], label='Lower Band', color='green')
    #     # plt.fill_between(df.index, df['Real Lower Band'], df['Real Upper Band'], color='gray', alpha=0.3)
    #     # plt.title(f'Bollinger Bands for {SYMBOL}')
    #     # plt.xlabel('Date')
    #     # plt.ylabel('Price')
    #     # plt.legend()
    #     # plt.grid(True)
    #     # plt.show()
    # else:
    #     print("\nCould not find 'Technical Analysis: BBANDS' in the API response.")
    #     if "Error Message" in data:
    #         print(f"API Error: {data['Error Message']}")
    #     elif "Information" in data:
    #         print(f"API Info: {data['Information']}") # Often indicates API call limit reached

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