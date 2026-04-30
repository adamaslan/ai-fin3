"""
YFinance Signal Detector with Safe JSON/MD Export
Fetches data, calculates indicators, detects signals, exports to JSON and Markdown
Handles all edge cases for safe JSON serialization

New signals: Ichimoku Cloud, OBV/CMF divergence, support/resistance zones
New time intervals: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y + intraday 1m/5m/15m/30m/60m
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path


# Maps period strings to the interval yfinance should use for that period
PERIOD_INTERVAL_MAP = {
    '1d':   '1m',
    '5d':   '5m',
    '1mo':  '15m',
    '3mo':  '1h',
    '6mo':  '1d',
    '1y':   '1d',
    '2y':   '1d',
    '5y':   '1wk',
    'max':  '1mo',
}


class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles pandas/numpy types safely"""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass
        try:
            return str(obj)
        except Exception:
            return None


def _safe_float(value):
    """Module-level safe float helper; returns None on NaN/Inf."""
    try:
        if pd.isna(value) or np.isinf(float(value)):
            return None
        return float(value)
    except Exception:
        return None


def _safe_value(value):
    """Like _safe_float but falls back to 0.0 for JSON numeric fields."""
    result = _safe_float(value)
    return result if result is not None else 0.0


class SignalDetectorExporter:
    """Complete signal detection and export pipeline."""

    def __init__(self, symbol: str, period: str = '1y', output_dir: str = 'signal_reports'):
        self.symbol = symbol
        self.period = period
        self.interval = PERIOD_INTERVAL_MAP.get(period, '1d')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.data = None
        self.signals = []
        self.current = None

    # ------------------------------------------------------------------ #
    #  Data fetching                                                       #
    # ------------------------------------------------------------------ #

    def fetch_data(self) -> pd.DataFrame:
        """Fetch OHLCV data from yfinance using the appropriate interval."""
        print(f"📊 Fetching {self.symbol}  period={self.period}  interval={self.interval} ...")
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(period=self.period, interval=self.interval)

        if self.data.empty:
            raise ValueError(f"No data found for {self.symbol}")

        print(f"✅ Fetched {len(self.data)} bars")
        return self.data

    # ------------------------------------------------------------------ #
    #  Indicator calculation                                               #
    # ------------------------------------------------------------------ #

    def calculate_indicators(self) -> pd.DataFrame:
        """Calculate all technical indicators."""
        print("🔧 Calculating indicators...")
        df = self.data.copy()

        # Moving Averages
        for p in [10, 20, 50, 100, 200]:
            df[f'SMA_{p}'] = df['Close'].rolling(window=p).mean()
            df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()

        # RSI (three periods)
        for p in [9, 14, 21]:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=p).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=p).mean()
            rs = gain / loss
            df[f'RSI_{p}'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI_14']

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # Bollinger Bands (three widths)
        for p in [10, 20, 30]:
            mid = df['Close'].rolling(window=p).mean()
            std = df['Close'].rolling(window=p).std()
            df[f'BB_{p}_Upper'] = mid + std * 2
            df[f'BB_{p}_Lower'] = mid - std * 2
            denom = df[f'BB_{p}_Upper'] - df[f'BB_{p}_Lower']
            df[f'BB_{p}_Position'] = (df['Close'] - df[f'BB_{p}_Lower']) / denom

        # Stochastic
        low14 = df['Low'].rolling(window=14).min()
        high14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low14) / (high14 - low14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

        # ATR & ADX
        hl = df['High'] - df['Low']
        hc = np.abs(df['High'] - df['Close'].shift())
        lc = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()

        plus_dm = df['High'].diff().clip(lower=0)
        minus_dm = (-df['Low'].diff()).clip(lower=0)
        tr14 = true_range.rolling(14).sum()
        plus_di = 100 * (plus_dm.rolling(14).sum() / tr14)
        minus_di = 100 * (minus_dm.rolling(14).sum() / tr14)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(14).mean()
        df['Plus_DI'] = plus_di
        df['Minus_DI'] = minus_di

        # Volume indicators
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_EMA'] = df['OBV'].ewm(span=20, adjust=False).mean()

        # CMF (Chaikin Money Flow, 20 periods)
        mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        mfm = mfm.fillna(0)
        mfv = mfm * df['Volume']
        df['CMF'] = mfv.rolling(20).sum() / df['Volume'].rolling(20).sum()

        # Price changes
        df['Price_Change'] = df['Close'].pct_change() * 100
        df['Price_Change_5'] = df['Close'].pct_change(5) * 100

        # 52-bar high/low (proxy for 52-week when daily; adapts to interval)
        lookback = min(252, len(df))
        df['High_52w'] = df['High'].rolling(window=lookback).max()
        df['Low_52w'] = df['Low'].rolling(window=lookback).min()

        # Distance from key MAs
        for p in [20, 50, 200]:
            df[f'Dist_SMA_{p}'] = ((df['Close'] - df[f'SMA_{p}']) / df[f'SMA_{p}']) * 100

        # Ichimoku Cloud
        df = self._calculate_ichimoku(df)

        self.data = df
        print("✅ Indicators calculated")
        return df

    def _calculate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Ichimoku Cloud columns to df."""
        nine_high = df['High'].rolling(9).max()
        nine_low = df['Low'].rolling(9).min()
        df['Ichimoku_Tenkan'] = (nine_high + nine_low) / 2          # Conversion line

        period26_high = df['High'].rolling(26).max()
        period26_low = df['Low'].rolling(26).min()
        df['Ichimoku_Kijun'] = (period26_high + period26_low) / 2   # Base line

        df['Ichimoku_SpanA'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(26)
        period52_high = df['High'].rolling(52).max()
        period52_low = df['Low'].rolling(52).min()
        df['Ichimoku_SpanB'] = ((period52_high + period52_low) / 2).shift(26)
        df['Ichimoku_Chikou'] = df['Close'].shift(-26)               # Lagging span
        return df

    # ------------------------------------------------------------------ #
    #  Support / resistance zones                                         #
    # ------------------------------------------------------------------ #

    def _find_support_resistance(self, df: pd.DataFrame, window: int = 10, n_levels: int = 5) -> dict:
        """
        Identify support and resistance levels using local pivot highs/lows.
        Returns dict with 'support' and 'resistance' lists.
        """
        highs = df['High'].values
        lows = df['Low'].values
        close = df['Close'].values[-1]

        pivot_highs = []
        pivot_lows = []

        for i in range(window, len(df) - window):
            if highs[i] == max(highs[i - window:i + window + 1]):
                pivot_highs.append(highs[i])
            if lows[i] == min(lows[i - window:i + window + 1]):
                pivot_lows.append(lows[i])

        # Cluster nearby levels (within 0.5% of each other)
        def cluster(levels: list, tol: float = 0.005) -> list:
            if not levels:
                return []
            levels = sorted(set(levels))
            clusters = [[levels[0]]]
            for lvl in levels[1:]:
                if abs(lvl - clusters[-1][-1]) / clusters[-1][-1] <= tol:
                    clusters[-1].append(lvl)
                else:
                    clusters.append([lvl])
            return [np.mean(c) for c in clusters]

        support_levels = sorted(
            [lvl for lvl in cluster(pivot_lows) if lvl < close], reverse=True
        )[:n_levels]
        resistance_levels = sorted(
            [lvl for lvl in cluster(pivot_highs) if lvl > close]
        )[:n_levels]

        return {
            'support': [round(float(l), 4) for l in support_levels],
            'resistance': [round(float(l), 4) for l in resistance_levels],
        }

    # ------------------------------------------------------------------ #
    #  Signal detection helpers                                           #
    # ------------------------------------------------------------------ #

    def _detect_ichimoku_signals(self, df: pd.DataFrame, current: pd.Series, prev: pd.Series) -> list:
        signals = []
        cols = ['Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_SpanA', 'Ichimoku_SpanB']
        if any(col not in df.columns or pd.isna(current.get(col)) for col in cols):
            return signals

        tenkan = _safe_value(current['Ichimoku_Tenkan'])
        kijun = _safe_value(current['Ichimoku_Kijun'])
        span_a = _safe_value(current['Ichimoku_SpanA'])
        span_b = _safe_value(current['Ichimoku_SpanB'])
        close = _safe_value(current['Close'])
        cloud_top = max(span_a, span_b)
        cloud_bot = min(span_a, span_b)

        prev_tenkan = _safe_value(prev.get('Ichimoku_Tenkan', tenkan))
        prev_kijun = _safe_value(prev.get('Ichimoku_Kijun', kijun))

        # TK cross (bullish)
        if prev_tenkan <= prev_kijun and tenkan > kijun:
            signals.append({
                'signal': 'ICHIMOKU TK BULL CROSS',
                'description': f'Tenkan ({tenkan:.2f}) crossed above Kijun ({kijun:.2f})',
                'strength': 'STRONG BULLISH',
                'category': 'ICHIMOKU',
                'value': tenkan,
            })
        # TK cross (bearish)
        elif prev_tenkan >= prev_kijun and tenkan < kijun:
            signals.append({
                'signal': 'ICHIMOKU TK BEAR CROSS',
                'description': f'Tenkan ({tenkan:.2f}) crossed below Kijun ({kijun:.2f})',
                'strength': 'STRONG BEARISH',
                'category': 'ICHIMOKU',
                'value': tenkan,
            })

        # Price vs Cloud
        if close > cloud_top:
            signals.append({
                'signal': 'PRICE ABOVE KUMO',
                'description': f'Close ${close:.2f} above cloud top ${cloud_top:.2f}',
                'strength': 'BULLISH',
                'category': 'ICHIMOKU',
                'value': close - cloud_top,
            })
        elif close < cloud_bot:
            signals.append({
                'signal': 'PRICE BELOW KUMO',
                'description': f'Close ${close:.2f} below cloud bottom ${cloud_bot:.2f}',
                'strength': 'BEARISH',
                'category': 'ICHIMOKU',
                'value': cloud_bot - close,
            })
        else:
            signals.append({
                'signal': 'PRICE INSIDE KUMO',
                'description': f'Close ${close:.2f} inside cloud (indecision zone)',
                'strength': 'NEUTRAL',
                'category': 'ICHIMOKU',
                'value': close,
            })

        # Bullish cloud (SpanA > SpanB = green cloud)
        if span_a > span_b:
            signals.append({
                'signal': 'BULLISH KUMO',
                'description': f'Green cloud: SpanA ({span_a:.2f}) > SpanB ({span_b:.2f})',
                'strength': 'BULLISH',
                'category': 'ICHIMOKU',
                'value': span_a - span_b,
            })
        else:
            signals.append({
                'signal': 'BEARISH KUMO',
                'description': f'Red cloud: SpanA ({span_a:.2f}) < SpanB ({span_b:.2f})',
                'strength': 'BEARISH',
                'category': 'ICHIMOKU',
                'value': span_b - span_a,
            })

        return signals

    def _detect_obv_cmf_signals(self, df: pd.DataFrame, current: pd.Series, prev: pd.Series) -> list:
        signals = []

        # OBV divergence: price making new high but OBV not confirming
        if len(df) >= 20 and 'OBV' in df.columns and 'OBV_EMA' in df.columns:
            recent = df.tail(20)
            price_change = _safe_value(recent['Close'].iloc[-1]) - _safe_value(recent['Close'].iloc[0])
            obv_change = _safe_value(recent['OBV'].iloc[-1]) - _safe_value(recent['OBV'].iloc[0])

            if price_change > 0 and obv_change < 0:
                signals.append({
                    'signal': 'OBV BEARISH DIVERGENCE',
                    'description': f'Price rising but OBV falling (volume not confirming rally)',
                    'strength': 'BEARISH',
                    'category': 'OBV_CMF',
                    'value': _safe_value(current['OBV']),
                })
            elif price_change < 0 and obv_change > 0:
                signals.append({
                    'signal': 'OBV BULLISH DIVERGENCE',
                    'description': 'Price falling but OBV rising (smart money accumulating)',
                    'strength': 'STRONG BULLISH',
                    'category': 'OBV_CMF',
                    'value': _safe_value(current['OBV']),
                })

            # OBV trend confirmation
            obv_now = _safe_value(current.get('OBV'))
            obv_ema_now = _safe_value(current.get('OBV_EMA'))
            prev_obv = _safe_value(prev.get('OBV', obv_now))
            prev_obv_ema = _safe_value(prev.get('OBV_EMA', obv_ema_now))

            if obv_now is not None and obv_ema_now is not None:
                if prev_obv <= prev_obv_ema and obv_now > obv_ema_now:
                    signals.append({
                        'signal': 'OBV BULL CROSS EMA',
                        'description': 'OBV crossed above its 20-period EMA',
                        'strength': 'BULLISH',
                        'category': 'OBV_CMF',
                        'value': obv_now,
                    })
                elif prev_obv >= prev_obv_ema and obv_now < obv_ema_now:
                    signals.append({
                        'signal': 'OBV BEAR CROSS EMA',
                        'description': 'OBV crossed below its 20-period EMA',
                        'strength': 'BEARISH',
                        'category': 'OBV_CMF',
                        'value': obv_now,
                    })

        # CMF signals
        if 'CMF' in df.columns and not pd.isna(current.get('CMF')):
            cmf = _safe_value(current['CMF'])
            prev_cmf = _safe_value(prev.get('CMF', cmf))

            if cmf > 0.1:
                signals.append({
                    'signal': 'CMF STRONG BUYING PRESSURE',
                    'description': f'Chaikin Money Flow: {cmf:.3f} (strong accumulation)',
                    'strength': 'BULLISH',
                    'category': 'OBV_CMF',
                    'value': cmf,
                })
            elif cmf < -0.1:
                signals.append({
                    'signal': 'CMF STRONG SELLING PRESSURE',
                    'description': f'Chaikin Money Flow: {cmf:.3f} (strong distribution)',
                    'strength': 'BEARISH',
                    'category': 'OBV_CMF',
                    'value': cmf,
                })

            # CMF zero-line cross
            if prev_cmf is not None:
                if prev_cmf <= 0 and cmf > 0:
                    signals.append({
                        'signal': 'CMF CROSSED POSITIVE',
                        'description': f'CMF turned positive: {cmf:.3f}',
                        'strength': 'BULLISH',
                        'category': 'OBV_CMF',
                        'value': cmf,
                    })
                elif prev_cmf >= 0 and cmf < 0:
                    signals.append({
                        'signal': 'CMF CROSSED NEGATIVE',
                        'description': f'CMF turned negative: {cmf:.3f}',
                        'strength': 'BEARISH',
                        'category': 'OBV_CMF',
                        'value': cmf,
                    })

        return signals

    def _detect_sr_signals(self, df: pd.DataFrame, current: pd.Series, sr_zones: dict) -> list:
        """Generate signals when price is near a support or resistance level."""
        signals = []
        close = _safe_value(current['Close'])
        if close is None or close == 0:
            return signals

        proximity_pct = 0.015  # 1.5% proximity threshold

        for lvl in sr_zones.get('support', []):
            dist_pct = abs(close - lvl) / lvl
            if dist_pct <= proximity_pct:
                signals.append({
                    'signal': 'NEAR SUPPORT LEVEL',
                    'description': f'Price ${close:.2f} near support ${lvl:.2f} ({dist_pct*100:.1f}% away)',
                    'strength': 'BULLISH',
                    'category': 'SUPPORT_RESISTANCE',
                    'value': lvl,
                })

        for lvl in sr_zones.get('resistance', []):
            dist_pct = abs(close - lvl) / lvl
            if dist_pct <= proximity_pct:
                signals.append({
                    'signal': 'NEAR RESISTANCE LEVEL',
                    'description': f'Price ${close:.2f} near resistance ${lvl:.2f} ({dist_pct*100:.1f}% away)',
                    'strength': 'BEARISH',
                    'category': 'SUPPORT_RESISTANCE',
                    'value': lvl,
                })

        return signals

    # ------------------------------------------------------------------ #
    #  Main signal detection                                              #
    # ------------------------------------------------------------------ #

    def detect_signals(self) -> list:
        """Detect all technical signals."""
        print("🎯 Detecting signals...")

        df = self.data.copy()
        current = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else current
        self.current = current
        signals = []

        # ---- MA Crosses ----
        if len(df) > 200:
            if not pd.isna(current.get('SMA_50')) and not pd.isna(current.get('SMA_200')):
                if prev['SMA_50'] <= prev['SMA_200'] and current['SMA_50'] > current['SMA_200']:
                    signals.append({
                        'signal': 'GOLDEN CROSS',
                        'description': '50 MA crossed above 200 MA',
                        'strength': 'STRONG BULLISH',
                        'category': 'MA_CROSS',
                        'value': _safe_value(current['SMA_50']),
                    })
                elif prev['SMA_50'] >= prev['SMA_200'] and current['SMA_50'] < current['SMA_200']:
                    signals.append({
                        'signal': 'DEATH CROSS',
                        'description': '50 MA crossed below 200 MA',
                        'strength': 'STRONG BEARISH',
                        'category': 'MA_CROSS',
                        'value': _safe_value(current['SMA_50']),
                    })

        for fast, slow in [(10, 20), (20, 50)]:
            cf, cs = f'SMA_{fast}', f'SMA_{slow}'
            if cf in df.columns and cs in df.columns:
                if not pd.isna(current.get(cf)) and not pd.isna(current.get(cs)):
                    if prev[cf] <= prev[cs] and current[cf] > current[cs]:
                        signals.append({
                            'signal': f'{fast}/{slow} MA BULL CROSS',
                            'description': f'{fast} MA crossed above {slow} MA',
                            'strength': 'BULLISH',
                            'category': 'MA_CROSS',
                            'value': _safe_value(current[cf]),
                        })
                    elif prev[cf] >= prev[cs] and current[cf] < current[cs]:
                        signals.append({
                            'signal': f'{fast}/{slow} MA BEAR CROSS',
                            'description': f'{fast} MA crossed below {slow} MA',
                            'strength': 'BEARISH',
                            'category': 'MA_CROSS',
                            'value': _safe_value(current[cf]),
                        })

        # ---- RSI ----
        for p in [9, 14, 21]:
            col = f'RSI_{p}' if f'RSI_{p}' in df.columns else 'RSI'
            if col in df.columns and not pd.isna(current.get(col)):
                rsi = _safe_value(current[col])
                prev_rsi = _safe_value(prev.get(col, rsi))
                if rsi < 30:
                    signals.append({
                        'signal': f'RSI{p} OVERSOLD',
                        'description': f'RSI({p}): {rsi:.1f}',
                        'strength': 'BULLISH',
                        'category': 'RSI',
                        'value': rsi,
                    })
                elif rsi > 70:
                    signals.append({
                        'signal': f'RSI{p} OVERBOUGHT',
                        'description': f'RSI({p}): {rsi:.1f}',
                        'strength': 'BEARISH',
                        'category': 'RSI',
                        'value': rsi,
                    })
                # RSI midline cross
                if prev_rsi is not None:
                    if prev_rsi < 50 <= rsi:
                        signals.append({
                            'signal': f'RSI{p} CROSSED 50 BULL',
                            'description': f'RSI({p}) crossed above 50: {rsi:.1f}',
                            'strength': 'BULLISH',
                            'category': 'RSI',
                            'value': rsi,
                        })
                    elif prev_rsi > 50 >= rsi:
                        signals.append({
                            'signal': f'RSI{p} CROSSED 50 BEAR',
                            'description': f'RSI({p}) crossed below 50: {rsi:.1f}',
                            'strength': 'BEARISH',
                            'category': 'RSI',
                            'value': rsi,
                        })

        # ---- MACD ----
        if all(c in df.columns for c in ['MACD', 'MACD_Signal']):
            if not pd.isna(current.get('MACD')) and not pd.isna(current.get('MACD_Signal')):
                if prev['MACD'] <= prev['MACD_Signal'] and current['MACD'] > current['MACD_Signal']:
                    signals.append({
                        'signal': 'MACD BULL CROSS',
                        'description': 'MACD crossed above signal line',
                        'strength': 'STRONG BULLISH',
                        'category': 'MACD',
                        'value': _safe_value(current['MACD']),
                    })
                elif prev['MACD'] >= prev['MACD_Signal'] and current['MACD'] < current['MACD_Signal']:
                    signals.append({
                        'signal': 'MACD BEAR CROSS',
                        'description': 'MACD crossed below signal line',
                        'strength': 'STRONG BEARISH',
                        'category': 'MACD',
                        'value': _safe_value(current['MACD']),
                    })

        # ---- Bollinger Bands ----
        for bb_p in [10, 20, 30]:
            upper_col = f'BB_{bb_p}_Upper'
            lower_col = f'BB_{bb_p}_Lower'
            if upper_col in df.columns and lower_col in df.columns:
                if not pd.isna(current.get(upper_col)) and not pd.isna(current.get(lower_col)):
                    if current['Close'] > current[upper_col]:
                        signals.append({
                            'signal': f'ABOVE UPPER BB{bb_p}',
                            'description': f'Price broke above {bb_p}-period upper Bollinger Band',
                            'strength': 'EXTREME BULLISH',
                            'category': 'BB_BREAKOUT',
                            'value': _safe_value(current['Close'] - current[upper_col]),
                        })
                    elif current['Close'] < current[lower_col]:
                        signals.append({
                            'signal': f'BELOW LOWER BB{bb_p}',
                            'description': f'Price broke below {bb_p}-period lower Bollinger Band',
                            'strength': 'EXTREME BEARISH',
                            'category': 'BB_BREAKOUT',
                            'value': _safe_value(current[lower_col] - current['Close']),
                        })

        # ---- Volume ----
        if 'Volume_MA_20' in df.columns and not pd.isna(current.get('Volume_MA_20')):
            vol_ratio = _safe_value(current['Volume'] / current['Volume_MA_20'])
            if vol_ratio > 2:
                signals.append({
                    'signal': 'VOLUME SPIKE',
                    'description': f'Volume: {vol_ratio:.1f}x average',
                    'strength': 'SIGNIFICANT',
                    'category': 'VOLUME',
                    'value': vol_ratio,
                })
            elif vol_ratio < 0.5:
                signals.append({
                    'signal': 'LOW VOLUME',
                    'description': f'Volume: {vol_ratio:.1f}x average (weak conviction)',
                    'strength': 'NEUTRAL',
                    'category': 'VOLUME',
                    'value': vol_ratio,
                })

        # ---- Stochastic ----
        if 'Stoch_K' in df.columns and not pd.isna(current.get('Stoch_K')):
            stoch_k = _safe_value(current['Stoch_K'])
            stoch_d = _safe_value(current.get('Stoch_D', stoch_k))
            prev_k = _safe_value(prev.get('Stoch_K', stoch_k))
            prev_d = _safe_value(prev.get('Stoch_D', stoch_d))
            if stoch_k < 20:
                signals.append({
                    'signal': 'STOCHASTIC OVERSOLD',
                    'description': f'%K: {stoch_k:.1f}',
                    'strength': 'BULLISH',
                    'category': 'STOCHASTIC',
                    'value': stoch_k,
                })
            elif stoch_k > 80:
                signals.append({
                    'signal': 'STOCHASTIC OVERBOUGHT',
                    'description': f'%K: {stoch_k:.1f}',
                    'strength': 'BEARISH',
                    'category': 'STOCHASTIC',
                    'value': stoch_k,
                })
            # K/D cross
            if stoch_d is not None and prev_k is not None and prev_d is not None:
                if prev_k <= prev_d and stoch_k > stoch_d and stoch_k < 30:
                    signals.append({
                        'signal': 'STOCH BULL CROSS (OVERSOLD)',
                        'description': f'%K ({stoch_k:.1f}) crossed above %D in oversold zone',
                        'strength': 'STRONG BULLISH',
                        'category': 'STOCHASTIC',
                        'value': stoch_k,
                    })
                elif prev_k >= prev_d and stoch_k < stoch_d and stoch_k > 70:
                    signals.append({
                        'signal': 'STOCH BEAR CROSS (OVERBOUGHT)',
                        'description': f'%K ({stoch_k:.1f}) crossed below %D in overbought zone',
                        'strength': 'STRONG BEARISH',
                        'category': 'STOCHASTIC',
                        'value': stoch_k,
                    })

        # ---- ADX Trend ----
        if 'ADX' in df.columns and not pd.isna(current.get('ADX')):
            adx = _safe_value(current['ADX'])
            direction = 'UP' if current['Plus_DI'] > current['Minus_DI'] else 'DOWN'
            if adx > 40:
                signals.append({
                    'signal': f'VERY STRONG {direction}TREND',
                    'description': f'ADX: {adx:.1f} (very strong trend)',
                    'strength': 'TRENDING',
                    'category': 'ADX',
                    'value': adx,
                })
            elif adx > 25:
                signals.append({
                    'signal': f'STRONG {direction}TREND',
                    'description': f'ADX: {adx:.1f}',
                    'strength': 'TRENDING',
                    'category': 'ADX',
                    'value': adx,
                })

        # ---- Price Action ----
        if 'Price_Change' in df.columns and not pd.isna(current.get('Price_Change')):
            pc = _safe_value(current['Price_Change'])
            if pc > 5:
                signals.append({
                    'signal': 'LARGE GAIN',
                    'description': f'+{pc:.1f}% today',
                    'strength': 'STRONG BULLISH',
                    'category': 'PRICE_ACTION',
                    'value': pc,
                })
            elif pc < -5:
                signals.append({
                    'signal': 'LARGE LOSS',
                    'description': f'{pc:.1f}% today',
                    'strength': 'STRONG BEARISH',
                    'category': 'PRICE_ACTION',
                    'value': pc,
                })

        # ---- 52-bar High/Low ----
        if all(c in df.columns for c in ['High_52w', 'Low_52w']):
            if not pd.isna(current.get('High_52w')) and not pd.isna(current.get('Low_52w')):
                if current['Close'] >= current['High_52w'] * 0.999:
                    signals.append({
                        'signal': '52-WEEK HIGH',
                        'description': f'At ${_safe_value(current["Close"]):.2f}',
                        'strength': 'EXTREME BULLISH',
                        'category': 'RANGE',
                        'value': _safe_value(current['Close']),
                    })
                elif current['Close'] <= current['Low_52w'] * 1.001:
                    signals.append({
                        'signal': '52-WEEK LOW',
                        'description': f'At ${_safe_value(current["Close"]):.2f}',
                        'strength': 'EXTREME BEARISH',
                        'category': 'RANGE',
                        'value': _safe_value(current['Close']),
                    })

        # ---- Distance from MAs ----
        for p in [20, 50, 200]:
            dist_col = f'Dist_SMA_{p}'
            if dist_col in df.columns and not pd.isna(current.get(dist_col)):
                dist = _safe_value(current[dist_col])
                if dist > 10:
                    signals.append({
                        'signal': f'OVEREXTENDED ABOVE {p}MA',
                        'description': f'{dist:.1f}% above {p}MA',
                        'strength': 'BEARISH',
                        'category': 'MA_DISTANCE',
                        'value': dist,
                    })
                elif dist < -10:
                    signals.append({
                        'signal': f'OVEREXTENDED BELOW {p}MA',
                        'description': f'{abs(dist):.1f}% below {p}MA',
                        'strength': 'BULLISH',
                        'category': 'MA_DISTANCE',
                        'value': dist,
                    })

        # ---- Ichimoku Cloud ----
        signals.extend(self._detect_ichimoku_signals(df, current, prev))

        # ---- OBV / CMF Divergence ----
        signals.extend(self._detect_obv_cmf_signals(df, current, prev))

        # ---- Support / Resistance proximity ----
        sr_zones = self._find_support_resistance(df)
        signals.extend(self._detect_sr_signals(df, current, sr_zones))

        self.signals = signals
        self._sr_zones = sr_zones
        print(f"✅ Detected {len(signals)} signals")
        return signals

    # ------------------------------------------------------------------ #
    #  Export                                                              #
    # ------------------------------------------------------------------ #

    def export_json(self) -> Path:
        """Export signals and data to JSON."""
        print("💾 Exporting to JSON...")
        current = self.current
        sr = getattr(self, '_sr_zones', {'support': [], 'resistance': []})

        data = {
            'metadata': {
                'symbol': self.symbol,
                'timestamp': datetime.now().isoformat(),
                'data_period': self.period,
                'interval': self.interval,
                'total_bars': len(self.data),
                'signals_detected': len(self.signals),
            },
            'current_price': {
                'close': _safe_float(current['Close']),
                'open': _safe_float(current['Open']),
                'high': _safe_float(current['High']),
                'low': _safe_float(current['Low']),
                'volume': int(current['Volume']) if not pd.isna(current['Volume']) else 0,
                'change_pct': _safe_float(current.get('Price_Change', 0)),
            },
            'indicators': {
                'RSI': _safe_float(current.get('RSI')),
                'MACD': _safe_float(current.get('MACD')),
                'MACD_Signal': _safe_float(current.get('MACD_Signal')),
                'MACD_Hist': _safe_float(current.get('MACD_Hist')),
                'ADX': _safe_float(current.get('ADX')),
                'Stochastic_K': _safe_float(current.get('Stoch_K')),
                'Stochastic_D': _safe_float(current.get('Stoch_D')),
                'ATR': _safe_float(current.get('ATR')),
                'BB_Position': _safe_float(current.get('BB_20_Position')),
                'Volume_Ratio': _safe_float(
                    current['Volume'] / current.get('Volume_MA_20', current['Volume'])
                ),
                'OBV': _safe_float(current.get('OBV')),
                'CMF': _safe_float(current.get('CMF')),
                'Ichimoku_Tenkan': _safe_float(current.get('Ichimoku_Tenkan')),
                'Ichimoku_Kijun': _safe_float(current.get('Ichimoku_Kijun')),
                'Ichimoku_SpanA': _safe_float(current.get('Ichimoku_SpanA')),
                'Ichimoku_SpanB': _safe_float(current.get('Ichimoku_SpanB')),
            },
            'moving_averages': {
                f'SMA_{p}': _safe_float(current.get(f'SMA_{p}')) for p in [10, 20, 50, 100, 200]
            } | {
                f'EMA_{p}': _safe_float(current.get(f'EMA_{p}')) for p in [10, 20, 50, 200]
            },
            'support_resistance': sr,
            'signals': self.signals,
            'signal_summary': {
                'total': len(self.signals),
                'bullish': sum(1 for s in self.signals if 'BULLISH' in s['strength']),
                'bearish': sum(1 for s in self.signals if 'BEARISH' in s['strength']),
                'neutral': sum(1 for s in self.signals if 'NEUTRAL' in s['strength']),
                'by_category': {},
            },
        }

        for sig in self.signals:
            cat = sig['category']
            data['signal_summary']['by_category'][cat] = (
                data['signal_summary']['by_category'].get(cat, 0) + 1
            )

        filename = self.output_dir / f"{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, cls=SafeJSONEncoder)

        print(f"✅ JSON saved: {filename}")
        return filename

    def export_markdown(self) -> Path:
        """Export signals to formatted Markdown report."""
        print("📝 Exporting to Markdown...")
        current = self.current
        sr = getattr(self, '_sr_zones', {'support': [], 'resistance': []})

        def safe_val(val, decimals: int = 2) -> str:
            try:
                if pd.isna(val) or np.isinf(float(val)):
                    return "N/A"
                return f"{float(val):.{decimals}f}"
            except Exception:
                return "N/A"

        bullish = sum(1 for s in self.signals if 'BULLISH' in s['strength'])
        bearish = sum(1 for s in self.signals if 'BEARISH' in s['strength'])
        neutral = len(self.signals) - bullish - bearish

        by_category: dict = {}
        for sig in self.signals:
            by_category[sig['category']] = by_category.get(sig['category'], 0) + 1

        signals_by_cat: dict = {}
        for sig in self.signals:
            signals_by_cat.setdefault(sig['category'], []).append(sig)

        sr_support_str = ' | '.join(f"${l:.2f}" for l in sr['support']) or 'None detected'
        sr_resist_str = ' | '.join(f"${l:.2f}" for l in sr['resistance']) or 'None detected'

        md = f"""# Technical Analysis Report: {self.symbol}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Period:** {self.period}  |  **Interval:** {self.interval}
**Total Signals:** {len(self.signals)}

---

## 📊 Current Price Data

| Metric | Value |
|--------|-------|
| **Close** | ${safe_val(current['Close'])} |
| **Open** | ${safe_val(current['Open'])} |
| **High** | ${safe_val(current['High'])} |
| **Low** | ${safe_val(current['Low'])} |
| **Volume** | {int(current['Volume']):,} |
| **Change %** | {safe_val(current.get('Price_Change', 0))}% |

---

## 📈 Technical Indicators

| Indicator | Value |
|-----------|-------|
| **RSI(14)** | {safe_val(current.get('RSI'))} |
| **MACD** | {safe_val(current.get('MACD'), 4)} |
| **MACD Signal** | {safe_val(current.get('MACD_Signal'), 4)} |
| **ADX** | {safe_val(current.get('ADX'))} |
| **Stochastic %K** | {safe_val(current.get('Stoch_K'))} |
| **Stochastic %D** | {safe_val(current.get('Stoch_D'))} |
| **ATR** | {safe_val(current.get('ATR'))} |
| **OBV** | {safe_val(current.get('OBV'), 0)} |
| **CMF** | {safe_val(current.get('CMF'), 4)} |
| **Ichimoku Tenkan** | {safe_val(current.get('Ichimoku_Tenkan'))} |
| **Ichimoku Kijun** | {safe_val(current.get('Ichimoku_Kijun'))} |
| **Ichimoku SpanA** | {safe_val(current.get('Ichimoku_SpanA'))} |
| **Ichimoku SpanB** | {safe_val(current.get('Ichimoku_SpanB'))} |

### Moving Averages

| Period | SMA | EMA |
|--------|-----|-----|
| **10** | {safe_val(current.get('SMA_10'))} | {safe_val(current.get('EMA_10'))} |
| **20** | {safe_val(current.get('SMA_20'))} | {safe_val(current.get('EMA_20'))} |
| **50** | {safe_val(current.get('SMA_50'))} | {safe_val(current.get('EMA_50'))} |
| **200** | {safe_val(current.get('SMA_200'))} | {safe_val(current.get('EMA_200'))} |

---

## 🗺️ Support & Resistance Zones

| Type | Levels |
|------|--------|
| **Support** | {sr_support_str} |
| **Resistance** | {sr_resist_str} |

---

## 🎯 Detected Signals ({len(self.signals)} Total)

| Strength | Count |
|----------|-------|
| 🟢 **Bullish** | {bullish} |
| 🔴 **Bearish** | {bearish} |
| ⚪ **Neutral** | {neutral} |

### Signals by Category

"""
        for cat, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
            md += f"- **{cat}**: {count} signal{'s' if count > 1 else ''}\n"

        md += "\n---\n\n### All Signals\n"

        for cat in sorted(signals_by_cat.keys()):
            md += f"\n#### {cat.replace('_', ' ').title()}\n\n"
            for sig in signals_by_cat[cat]:
                emoji = "🟢" if "BULLISH" in sig['strength'] else "🔴" if "BEARISH" in sig['strength'] else "⚪"
                md += f"**{emoji} {sig['signal']}**\n"
                md += f"- {sig['description']}\n"
                md += f"- Strength: {sig['strength']}\n"
                md += f"- Value: {safe_val(sig['value'])}\n\n"

        md += "\n---\n\n*Report generated by YFinance Signal Detector*\n"

        filename = self.output_dir / f"{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(filename, 'w') as f:
            f.write(md)

        print(f"✅ Markdown saved: {filename}")
        return filename

    def run_complete_analysis(self) -> dict:
        """Run complete pipeline: fetch → calculate → detect → export."""
        print(f"\n{'='*60}")
        print(f"📊 SIGNAL ANALYSIS: {self.symbol}  [{self.period} / {self.interval}]")
        print(f"{'='*60}\n")

        self.fetch_data()
        self.calculate_indicators()
        self.detect_signals()

        json_file = self.export_json()
        md_file = self.export_markdown()

        print(f"\n{'='*60}")
        print("✅ ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"JSON: {json_file}")
        print(f"MD:   {md_file}")
        print(f"Signals detected: {len(self.signals)}")
        print(f"{'='*60}\n")

        return {
            'json_file': json_file,
            'md_file': md_file,
            'signals': self.signals,
        }


# ============ MAIN EXECUTION ============

if __name__ == "__main__":
    # Single stock — daily bars, 1-year window
    print("\n=== SINGLE STOCK ANALYSIS ===\n")
    detector = SignalDetectorExporter(symbol="AAPL", period="1y")
    results = detector.run_complete_analysis()
    print(f"\nDetected {len(results['signals'])} signals")
    print(f"Files saved to: {detector.output_dir}")

    # Demonstrate multiple periods / intervals
    print("\n\n=== MULTI-PERIOD ANALYSIS ===\n")
    for period in ['6mo', '3mo', '1mo', '5d']:
        try:
            d = SignalDetectorExporter(symbol="AAPL", period=period)
            d.run_complete_analysis()
        except Exception as e:
            print(f"❌ {period}: {e}")

    # Batch
    print("\n\n=== BATCH ANALYSIS ===\n")
    for sym in ['MSFT', 'GOOGL', 'TSLA']:
        try:
            SignalDetectorExporter(symbol=sym, period="6mo").run_complete_analysis()
        except Exception as e:
            print(f"❌ Error with {sym}: {e}")

    print("\n✅ All analyses complete!")
