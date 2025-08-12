
import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import time
import warnings
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

# Basic script setup
warnings.filterwarnings('ignore')


# Cell 2: Options Delta Analyzer Class
class AlphaVantageOptionsDeltaAnalyzer:
    """
    Analyzes options delta. Core logic is retained for fetching prices and
    simulating option chains to calculate delta, as this is prerequisite for analysis.
    Non-delta calculations (Gamma, Theta) have been removed.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alpha-vantage.co/query"
        self.risk_free_rate = 0.05  # 5% risk-free rate
        self.rate_limit_delay = 12  # Delay for Alpha Vantage free tier

    def make_api_call(self, params):
        """Make API call with rate limiting and basic error handling."""
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            if "Error Message" in data:
                print(f"API Error: {data['Error Message']}")
                return None
            if "Note" in data:
                print(f"API Note: {data['Note']}") # Handles rate limit messages
                return None
            time.sleep(self.rate_limit_delay)
            return data
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None

    def get_current_price(self, symbol):
        """Get the current stock price from Alpha Vantage."""
        params = {'function': 'GLOBAL_QUOTE', 'symbol': symbol, 'apikey': self.api_key}
        data = self.make_api_call(params)
        if data and 'Global Quote' in data and data['Global Quote']:
            try:
                return float(data['Global Quote']['05. price'])
            except (KeyError, ValueError) as e:
                print(f"Error parsing current price for {symbol}: {e}")
        return None

    def simulate_options_data(self, symbol, current_price, price_range=20):
        """
        Simulates an options chain for delta calculation.
        NOTE: This is necessary because the Alpha Vantage free tier does not provide options data.
        This simulation generates a realistic set of options to analyze.
        """
        print("Note: Simulating options data for analysis.")
        options_data = []
        today = datetime.now().date()

        # Generate plausible expiration dates (e.g., upcoming Fridays)
        expiry_dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d')
                        for i in range(1, 30) if (today + timedelta(days=i)).weekday() == 4]
        if not expiry_dates: expiry_dates.append((today + timedelta(days=14)).strftime('%Y-%m-%d'))

        strike_increment = 1.0 if current_price < 100 else 2.5
        strikes = np.arange(current_price - price_range, current_price + price_range, strike_increment)

        for expiry in expiry_dates:
            expiry_date = datetime.strptime(expiry, '%Y-%m-%d').date()
            time_to_expiry = ((expiry_date - today).days) / 365.0
            if time_to_expiry <= 0: continue

            for strike in strikes:
                # Use a fixed volatility for simulation consistency
                volatility = 0.30
                for option_type in ['call', 'put']:
                    options_data.append({
                        'type': option_type,
                        'strike': strike,
                        'expiry': expiry,
                        'time_to_expiry': time_to_expiry,
                        'days_to_expiry': (expiry_date - today).days,
                        'implied_vol': volatility, # Simplified for simulation
                        'volume': int(100 * np.random.random()), # Dummy volume
                    })
        return pd.DataFrame(options_data)

    def black_scholes_delta(self, S, K, T, r, sigma, option_type='call'):
        """Calculates Black-Scholes delta. This is the core calculation."""
        if T <= 0 or sigma <= 0:
            # For expired or invalid options, delta is deterministic
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else: # put
                return -1.0 if S < K else 0.0
        try:
            d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
            if option_type == 'call':
                return norm.cdf(d1)
            else:  # put
                return norm.cdf(d1) - 1
        except (ValueError, ZeroDivisionError):
            return 0

    def analyze_options_delta(self, symbol, price_range=20):
        """Main analysis function to orchestrate data fetching and delta calculation."""
        print(f"\nAnalyzing options delta for {symbol}...")
        current_price = self.get_current_price(symbol)
        if not current_price:
            print(f"❌ Could not fetch current price for {symbol}. Aborting analysis.")
            return None, None

        print(f"✅ Current price for {symbol}: ${current_price:.2f}")
        options_df = self.simulate_options_data(symbol, current_price, price_range)

        if options_df.empty:
            print("No options data could be simulated.")
            return None, None

        # Calculate delta for each option
        options_df['delta'] = options_df.apply(
            lambda row: self.black_scholes_delta(
                current_price, row['strike'], row['time_to_expiry'],
                self.risk_free_rate, row['implied_vol'], row['type']
            ), axis=1
        )

        # Create a simplified summary dictionary with current price
        summary = {'symbol': symbol, 'current_price': current_price}
        return options_df, summary

    def create_delta_summary(self, options_df, summary):
        """Creates summary statistics focused purely on delta."""
        if options_df is None or options_df.empty:
            return None

        calls = options_df[options_df['type'] == 'call']
        puts = options_df[options_df['type'] == 'put']

        summary.update({
            'total_options': len(options_df),
            'avg_call_delta': calls['delta'].mean(),
            'avg_put_delta': puts['delta'].mean(),
            'max_call_delta': calls['delta'].max(),
            'min_put_delta': puts['delta'].min(),
            'high_delta_calls': len(calls[calls['delta'] > 0.7]),
            'high_delta_puts': len(puts[puts['delta'] < -0.7]),
            'delta_range': options_df['delta'].max() - options_df['delta'].min(),
            'expiry_dates': sorted(options_df['expiry'].unique()),
            'strike_range': f"${options_df['strike'].min():.2f} - ${options_df['strike'].max():.2f}"
        })
        return summary


# Cell 3: Delta Strategy and Risk Classes
class DeltaNeutralStrategies:
    """Analyzes and suggests delta-based trading strategies."""
    def __init__(self, options_data, current_price):
        self.options_data = options_data
        self.current_price = current_price

    def find_delta_neutral_combinations(self, tolerance=0.1):
        """Finds option pairs that approximate a target delta."""
        if self.options_data is None or self.options_data.empty: return []

        combinations = []
        calls = self.options_data[self.options_data['type'] == 'call']
        puts = self.options_data[self.options_data['type'] == 'put']

        for _, call in calls.iterrows():
            # Find puts with the same expiry to form a synthetic position
            matching_puts = puts[puts['expiry'] == call['expiry']]
            for _, put in matching_puts.iterrows():
                # Assumes shorting the put to create a synthetic long stock
                combined_delta = call['delta'] - put['delta']
                if abs(combined_delta - 1) <= tolerance: # Synthetic long stock delta is ~1
                     combinations.append({
                        'strategy': 'Synthetic Long Stock',
                        'call_strike': call['strike'],
                        'put_strike': put['strike'],
                        'expiry': call['expiry'],
                        'combined_delta': combined_delta,
                     })
        return sorted(combinations, key=lambda x: abs(x['combined_delta']-1))

    def suggest_hedge_ratio(self, current_position_delta, target_delta=0):
        """Suggests options to hedge a position to a target delta."""
        delta_to_hedge = target_delta - current_position_delta
        hedge_suggestions = []

        for _, option in self.options_data.iterrows():
            if option['delta'] != 0:
                # Required contracts to offset the delta
                required_contracts = -delta_to_hedge / (option['delta'] * 100)
                hedge_suggestions.append({
                    'option_type': 'BUY '+option['type'].upper() if required_contracts > 0 else 'SELL '+option['type'].upper(),
                    'strike': option['strike'],
                    'expiry': option['expiry'],
                    'option_delta': option['delta'],
                    'required_contracts': round(required_contracts, 2)
                })
        # Return top 5 suggestions based on lowest number of contracts needed
        return sorted(hedge_suggestions, key=lambda x: abs(x['required_contracts']))[:5]


class DeltaRiskManager:
    """Risk management tools focused on delta exposure."""
    def __init__(self, options_data, current_price):
        self.options_data = options_data
        self.current_price = current_price

    def calculate_var_delta(self, confidence_level=0.95, time_horizon_days=1):
        """
        Calculates Value at Risk (VaR) for positions based only on delta.
        The contribution from Gamma has been removed to focus the analysis.
        """
        if self.options_data is None or self.options_data.empty: return None

        daily_vol = self.options_data['implied_vol'].mean() / np.sqrt(252)
        z_score = norm.ppf(confidence_level)
        price_move = self.current_price * daily_vol * z_score * np.sqrt(time_horizon_days)

        var_estimates = []
        for _, option in self.options_data.iterrows():
            # P&L is estimated using only delta's impact
            delta_pnl = option['delta'] * price_move * 100  # For one contract
            var_estimates.append({
                'option_type': option['type'],
                'strike': option['strike'],
                'expiry': option['expiry'],
                'delta': option['delta'],
                'potential_price_move': price_move,
                'var_estimate_pnl': delta_pnl
            })
        return pd.DataFrame(var_estimates)

    def monitor_delta_decay(self):
        """
        Analyzes how delta is expected to change as time passes (e.g., in one day).
        The reference to Theta has been removed.
        """
        if self.options_data is None or self.options_data.empty: return None

        decay_analysis = []
        analyzer = AlphaVantageOptionsDeltaAnalyzer(ALPHA_VANTAGE_API_KEY)

        for _, option in self.options_data.iterrows():
            future_tte = max(0, option['time_to_expiry'] - 1/365.0)

            # Recalculate delta for T-1 day
            future_delta = analyzer.black_scholes_delta(
                self.current_price, option['strike'], future_tte,
                0.05, option['implied_vol'], option['type']
            )

            decay_analysis.append({
                'option_type': option['type'],
                'strike': option['strike'],
                'expiry': option['expiry'],
                'current_delta': option['delta'],
                'expected_delta_next_day': future_delta,
                'delta_decay': future_delta - option['delta'],
                'days_to_expiry': option['days_to_expiry'],
            })
        return pd.DataFrame(decay_analysis)


# Cell 4: Main Execution
def main_delta_analysis_demo():
    """Comprehensive demo of the focused delta analysis."""
    if not ALPHA_VANTAGE_API_KEY:
        print("❌ API key not found. Please set ALPHA_VANTAGE_API_KEY in your .env file.")
        return

    print("🚀 Starting Focused Options Delta Analysis Demo")
    analyzer = AlphaVantageOptionsDeltaAnalyzer(ALPHA_VANTAGE_API_KEY)

    # 1. Single Symbol Analysis
    symbol_to_analyze = 'TSLA'
    options_data, summary = analyzer.analyze_options_delta(symbol_to_analyze)

    if options_data is not None:
        full_summary = analyzer.create_delta_summary(options_data, summary)
        print("\n--- Delta Summary ---")
        for key, val in full_summary.items():
            if isinstance(val, float): print(f"{key.replace('_', ' ').title()}: {val:.3f}")
            else: print(f"{key.replace('_', ' ').title()}: {val}")

        # 2. Delta Neutral Strategies
        print("\n--- Delta Strategy Analysis ---")
        strategy_analyzer = DeltaNeutralStrategies(options_data, summary['current_price'])
        neutral_combos = strategy_analyzer.find_delta_neutral_combinations(tolerance=0.1)
        if neutral_combos:
            print("\nTop 5 Synthetic Long Stock Combinations (Delta ≈ 1.0):")
            df = pd.DataFrame(neutral_combos[:5])
            print(df.to_string(index=False))
        else:
            print("No suitable delta neutral combinations found.")

        # 3. Delta Hedging
        print("\n--- Delta Hedging Example ---")
        # Example: You own 100 shares of TSLA (delta = +100) and want to be delta-neutral
        hedge_suggestions = strategy_analyzer.suggest_hedge_ratio(current_position_delta=100, target_delta=0)
        print("Suggestions to hedge 100 shares of TSLA to neutral:")
        df = pd.DataFrame(hedge_suggestions)
        print(df.to_string(index=False))

        # 4. Risk Management (VaR and Decay)
        print("\n--- Delta Risk Management ---")
        risk_manager = DeltaRiskManager(options_data, summary['current_price'])

        # VaR Analysis based on Delta
        var_analysis = risk_manager.calculate_var_delta()
        if var_analysis is not None:
            print("\nTop 5 Options with Highest Delta VaR (Potential P&L Swing):")
            # Sort by absolute P&L change
            top_var = var_analysis.reindex(var_analysis['var_estimate_pnl'].abs().sort_values(ascending=False).index)
            print(top_var.head(5).round(4).to_string(index=False))

        # Delta Decay Monitoring
        decay_analysis = risk_manager.monitor_delta_decay()
        if decay_analysis is not None:
            print("\nOptions with Highest Expected Delta Change (Next 24h):")
            high_decay = decay_analysis.reindex(decay_analysis['delta_decay'].abs().sort_values(ascending=False).index)
            print(high_decay[['option_type', 'strike', 'current_delta', 'expected_delta_next_day', 'delta_decay']].head(5).round(4).to_string(index=False))

    print(f"\n{'='*60}")
    print("✅ Analysis Complete!")
    print("\n📊 Key Takeaways:")
    print("- Delta measures an option's price change for a $1 change in the underlying stock.")
    print("- Call deltas are positive (0 to 1), Put deltas are negative (-1 to 0).")
    print("- At-the-money options have deltas around ±0.5.")
    print("- Delta can be used to construct neutral positions or hedge existing risk.")


if __name__ == "__main__":
    main_delta_analysis_demo()

