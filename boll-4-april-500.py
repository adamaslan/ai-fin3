"""
YFinance Signal Detector — April 500 Edition
Massively expanded signal detection: ~466 signals/bar across all categories.
Fixes from code review:
  - S/R proximity resolved to 0.5% (matching cluster tolerance)
  - Zero-division guard on S/R level
  - Neutral count in markdown uses explicit 'NEUTRAL' match; adds 'other' row
  - EMA_100 included in JSON moving_averages export
  - Price_Change_5 exported in JSON current_price block
  - S/R pivot pre-computation reduces detect_signals from O(N^2) to O(N)
  - NaN volume guard in markdown export
  - Single shared timestamp across all export filenames/headers
Historical bar scanning: evaluates every bar, not just the last one.
Confluence ranking: aggregates signals into net bias + confidence per bar.
Multi-timeframe analysis: combines scores across intervals for day/swing/invest views.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path


PERIOD_INTERVAL_MAP = {
    '1d':  '1m',
    '5d':  '5m',
    '1mo': '15m',
    '3mo': '1h',
    '6mo': '1d',
    '1y':  '1d',
    '2y':  '1d',
    '5y':  '1wk',
    'max': '1mo',
}

# Expanded parameter sets
MA_FAST_SLOW_PAIRS = [
    (5, 10), (5, 20), (5, 50), (10, 20), (10, 50), (10, 100),
    (20, 50), (20, 100), (20, 200), (50, 100), (50, 200),
]
RSI_PERIODS = [5, 10, 14, 20, 30, 40, 50]
RSI_OS_OB_LEVELS = [(20, 80), (25, 75), (30, 70), (35, 65)]
MACD_PARAMS = [(12, 26, 9), (10, 20, 5), (19, 39, 9), (20, 50, 10)]
BB_PERIODS = [10, 20, 30, 50]
BB_STD_DEVS = [1.5, 2.0, 2.5, 3.0]
VOLUME_MAS = [5, 10, 20, 50]
VOLUME_SPIKE_THRESHOLDS = [1.5, 2.0, 3.0]
VOLUME_LOW_THRESHOLDS = [0.3, 0.5, 0.7]
PRICE_ACTION_LOOKBACKS = [1, 5, 10, 20]
PRICE_ACTION_THRESHOLDS = [3, 5, 7, 10]
HL_LOOKBACKS = [20, 50, 100, 200, 252]
HL_PROXIMITIES = [0.01, 0.02, 0.05]
MA_DIST_PERIODS = [5, 10, 20, 30, 50, 100, 150, 200]
MA_DIST_THRESHOLDS = [5, 10, 15, 20]
SR_PIVOT_WINDOWS = [5, 10, 15, 20]
SR_PROXIMITIES = [0.005, 0.01, 0.015, 0.02]


class SafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles pandas/numpy types safely."""

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
    """Returns None on NaN/Inf."""
    try:
        if pd.isna(value) or np.isinf(float(value)):
            return None
        return float(value)
    except Exception:
        return None


def _safe_value(value):
    """Like _safe_float but falls back to 0.0."""
    result = _safe_float(value)
    return result if result is not None else 0.0


class ConfluenceRanker:
    """Aggregates a list of signals into a net bias and confidence score."""

    STRENGTH_SCORES: dict[str, float] = {
        'EXTREME BULLISH': 3.0,
        'STRONG BULLISH':  2.0,
        'BULLISH':         1.0,
        'TRENDING':        0.5,   # direction resolved from signal name
        'NEUTRAL':         0.0,
        'SIGNIFICANT':     0.0,
        'BEARISH':        -1.0,
        'STRONG BEARISH': -2.0,
        'EXTREME BEARISH':-3.0,
    }

    @classmethod
    def score_signal(cls, signal: dict) -> float:
        """Return numeric score, adjusting TRENDING by embedded direction."""
        strength = signal['strength']
        score = cls.STRENGTH_SCORES.get(strength, 0.0)
        if strength == 'TRENDING':
            name = signal['signal']
            score = 0.5 if 'UPTREND' in name else (-0.5 if 'DOWNTREND' in name else 0.0)
        return score

    @classmethod
    def rank_signals(cls, signals: list[dict]) -> dict:
        """Compute per-bar confluence statistics from a list of signal dicts.

        Returns a dict with:
          bullish_score, bearish_score, net_score, bias, confidence,
          signal_count, agreement_ratio, bullish_signal_count, bearish_signal_count
        """
        bullish_score = 0.0
        bearish_score = 0.0
        bullish_count = 0
        bearish_count = 0

        for sig in signals:
            s = cls.score_signal(sig)
            if s > 0:
                bullish_score += s
                bullish_count += 1
            elif s < 0:
                bearish_score += abs(s)
                bearish_count += 1

        net = bullish_score - bearish_score
        total = bullish_count + bearish_count

        if net > 0.5:
            bias = 'BULLISH'
        elif net < -0.5:
            bias = 'BEARISH'
        else:
            bias = 'NEUTRAL'

        abs_net = abs(net)
        winning_side = bullish_count if net >= 0 else bearish_count
        if abs_net >= 5.0 and (total == 0 or winning_side > total / 2):
            confidence = 'HIGH'
        elif abs_net >= 2.0:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        agreement = bullish_count / total if total > 0 else 0.5

        return {
            'bullish_score': round(bullish_score, 2),
            'bearish_score': round(bearish_score, 2),
            'net_score': round(net, 2),
            'bias': bias,
            'confidence': confidence,
            'signal_count': total,
            'agreement_ratio': round(agreement, 3),
            'bullish_signal_count': bullish_count,
            'bearish_signal_count': bearish_count,
        }

    @staticmethod
    def interpret(rank: dict) -> str:
        """One-sentence human interpretation of a confluence rank."""
        bias = rank['bias']
        conf = rank['confidence']
        net = rank['net_score']
        ar = rank['agreement_ratio']
        pct = int(ar * 100)
        sign = '+' if net >= 0 else ''
        return (
            f"{conf} confidence {bias} bias (net score {sign}{net:.1f}; "
            f"{pct}% of directional signals agree)."
        )


TIMEFRAME_CONFIGS: list[dict] = [
    {'label': '5m',     'period': '5d',  'interval': '5m',   'weight': 0.10, 'use_case': 'Scalp / day entry'},
    {'label': '15m',    'period': '1mo', 'interval': '15m',  'weight': 0.15, 'use_case': 'Day trend'},
    {'label': '1h',     'period': '3mo', 'interval': '1h',   'weight': 0.25, 'use_case': 'Swing trade'},
    {'label': 'Daily',  'period': '1y',  'interval': '1d',   'weight': 0.30, 'use_case': 'Position / invest'},
    {'label': 'Weekly', 'period': '5y',  'interval': '1wk',  'weight': 0.20, 'use_case': 'Long-term trend'},
]


class SignalDetectorExporter:
    """Complete signal detection and export pipeline — April 500 edition."""

    def __init__(self, symbol: str, period: str = '1y', output_dir: str = 'signal_reports'):
        self.symbol = symbol
        self.period = period
        self.interval = PERIOD_INTERVAL_MAP.get(period, '1d')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.data: pd.DataFrame | None = None
        self.signals: list[dict] = []            # latest-bar signals (backward compat)
        self.historical_signals: list[dict] = [] # all bars
        self.bar_confluence: list[dict] = []     # one confluence dict per scanned bar
        self.current: pd.Series | None = None

    # ------------------------------------------------------------------ #
    #  Data fetching                                                       #
    # ------------------------------------------------------------------ #

    def fetch_data(self) -> pd.DataFrame:
        """Fetch OHLCV data from yfinance using the appropriate interval."""
        print(f"Fetching {self.symbol}  period={self.period}  interval={self.interval} ...")
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(period=self.period, interval=self.interval)

        if self.data.empty:
            raise ValueError(f"No data found for {self.symbol}")

        print(f"Fetched {len(self.data)} bars")
        return self.data

    # ------------------------------------------------------------------ #
    #  Indicator calculation                                               #
    # ------------------------------------------------------------------ #

    def calculate_indicators(self) -> pd.DataFrame:
        """Calculate all technical indicators across expanded parameter sets."""
        print("Calculating indicators...")
        df = self.data.copy()

        # Moving Averages — all periods needed by detection
        all_ma_periods = sorted({p for pair in MA_FAST_SLOW_PAIRS for p in pair}
                                 | set(MA_DIST_PERIODS) | {50, 100, 200})
        for p in all_ma_periods:
            df[f'SMA_{p}'] = df['Close'].rolling(window=p).mean()
            df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()

        # RSI — all periods
        for p in RSI_PERIODS:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=p).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=p).mean()
            rs = gain / loss
            df[f'RSI_{p}'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI_14']

        # MACD — multiple parameter sets
        for fast, slow, sig in MACD_PARAMS:
            tag = f'_{fast}_{slow}_{sig}'
            exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
            exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
            df[f'MACD{tag}'] = exp1 - exp2
            df[f'MACD_Signal{tag}'] = df[f'MACD{tag}'].ewm(span=sig, adjust=False).mean()
            df[f'MACD_Hist{tag}'] = df[f'MACD{tag}'] - df[f'MACD_Signal{tag}']
        # canonical alias
        df['MACD'] = df['MACD_12_26_9']
        df['MACD_Signal'] = df['MACD_Signal_12_26_9']
        df['MACD_Hist'] = df['MACD_Hist_12_26_9']

        # Bollinger Bands — multiple periods × std devs
        for p in BB_PERIODS:
            mid = df['Close'].rolling(window=p).mean()
            std = df['Close'].rolling(window=p).std()
            for sd in BB_STD_DEVS:
                sd_tag = str(sd).replace('.', '_')
                df[f'BB_{p}_{sd_tag}_Upper'] = mid + std * sd
                df[f'BB_{p}_{sd_tag}_Lower'] = mid - std * sd
                denom = df[f'BB_{p}_{sd_tag}_Upper'] - df[f'BB_{p}_{sd_tag}_Lower']
                df[f'BB_{p}_{sd_tag}_Pct'] = np.where(
                    denom != 0,
                    (df['Close'] - df[f'BB_{p}_{sd_tag}_Lower']) / denom,
                    np.nan,
                )
                df[f'BB_{p}_{sd_tag}_Width'] = np.where(
                    mid != 0, denom / mid, np.nan
                )
        # canonical alias (20-period, 2σ)
        df['BB_20_Upper'] = df['BB_20_2_0_Upper']
        df['BB_20_Lower'] = df['BB_20_2_0_Lower']
        df['BB_20_Position'] = df['BB_20_2_0_Pct']

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

        # Volume MAs — multiple periods
        for vm in VOLUME_MAS:
            df[f'Volume_MA_{vm}'] = df['Volume'].rolling(window=vm).mean()

        # OBV & CMF
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_EMA'] = df['OBV'].ewm(span=20, adjust=False).mean()

        mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        mfm = mfm.fillna(0)
        mfv = mfm * df['Volume']
        df['CMF'] = mfv.rolling(20).sum() / df['Volume'].rolling(20).sum()

        # Price changes — multiple lookbacks
        for lb in PRICE_ACTION_LOOKBACKS:
            df[f'Price_Change_{lb}'] = df['Close'].pct_change(lb) * 100
        df['Price_Change'] = df['Price_Change_1']

        # Rolling high/low — all lookbacks
        for lb in HL_LOOKBACKS:
            actual_lb = min(lb, len(df))
            df[f'High_{lb}b'] = df['High'].rolling(window=actual_lb).max()
            df[f'Low_{lb}b'] = df['Low'].rolling(window=actual_lb).min()

        # Distance from SMAs — all periods
        for p in MA_DIST_PERIODS:
            sma_col = f'SMA_{p}'
            if sma_col in df.columns:
                df[f'Dist_SMA_{p}'] = np.where(
                    df[sma_col] != 0,
                    ((df['Close'] - df[sma_col]) / df[sma_col]) * 100,
                    np.nan,
                )

        # Ichimoku Cloud
        df = self._calculate_ichimoku(df)

        self.data = df
        self._precompute_pivots()
        print("Indicators calculated")
        return df

    def _calculate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Ichimoku Cloud columns to df."""
        nine_high = df['High'].rolling(9).max()
        nine_low = df['Low'].rolling(9).min()
        df['Ichimoku_Tenkan'] = (nine_high + nine_low) / 2

        period26_high = df['High'].rolling(26).max()
        period26_low = df['Low'].rolling(26).min()
        df['Ichimoku_Kijun'] = (period26_high + period26_low) / 2

        df['Ichimoku_SpanA'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(26)
        period52_high = df['High'].rolling(52).max()
        period52_low = df['Low'].rolling(52).min()
        df['Ichimoku_SpanB'] = ((period52_high + period52_low) / 2).shift(26)
        df['Ichimoku_Chikou'] = df['Close'].shift(-26)
        return df

    # ------------------------------------------------------------------ #
    #  Support / resistance zones                                         #
    # ------------------------------------------------------------------ #

    def _precompute_pivots(self) -> None:
        """Pre-calculate pivot highs/lows for every window in SR_PIVOT_WINDOWS.

        Runs once after calculate_indicators(). Stores results in
        self._pivot_cache[window] = {'highs': [...(bar_idx, price)], 'lows': [...]}
        so that _find_support_resistance can do an O(1) prefix-lookup instead of
        an O(N) scan per bar, reducing overall detect_signals complexity from
        O(N^2) to O(N).
        """
        df = self.data
        highs = df['High'].values
        lows = df['Low'].values
        n = len(df)

        self._pivot_cache: dict[int, dict] = {}
        for window in SR_PIVOT_WINDOWS:
            ph: list[tuple[int, float]] = []
            pl: list[tuple[int, float]] = []
            for i in range(window, n - window):
                lo = i - window
                hi = i + window + 1
                if highs[i] == highs[lo:hi].max():
                    ph.append((i, float(highs[i])))
                if lows[i] == lows[lo:hi].min():
                    pl.append((i, float(lows[i])))
            self._pivot_cache[window] = {'highs': ph, 'lows': pl}

    @staticmethod
    def _cluster_levels(levels: list[float], tol: float = 0.005) -> list[float]:
        if not levels:
            return []
        levels = sorted(set(levels))
        clusters: list[list[float]] = [[levels[0]]]
        for lvl in levels[1:]:
            ref = clusters[-1][-1]
            if ref != 0 and abs(lvl - ref) / ref <= tol:
                clusters[-1].append(lvl)
            else:
                clusters.append([lvl])
        return [float(np.mean(c)) for c in clusters]

    def _find_support_resistance(
        self, df: pd.DataFrame, window: int = 10, n_levels: int = 5
    ) -> dict:
        """Return clustered support/resistance levels up to bar len(df)-1.

        Uses the pre-computed pivot cache when available (O(k) prefix scan where
        k is the number of pivots), falling back to a full scan if the cache is
        absent (e.g., called standalone after detect_signals).
        """
        close = float(df['Close'].values[-1])
        bar_limit = len(df) - 1  # only use pivots up to this bar index

        if hasattr(self, '_pivot_cache') and window in self._pivot_cache:
            cache = self._pivot_cache[window]
            pivot_highs = [p for idx, p in cache['highs'] if idx < bar_limit]
            pivot_lows = [p for idx, p in cache['lows'] if idx < bar_limit]
        else:
            highs = df['High'].values
            lows = df['Low'].values
            pivot_highs = []
            pivot_lows = []
            for i in range(window, len(df) - window):
                lo, hi = i - window, i + window + 1
                if highs[i] == highs[lo:hi].max():
                    pivot_highs.append(float(highs[i]))
                if lows[i] == lows[lo:hi].min():
                    pivot_lows.append(float(lows[i]))

        support_levels = sorted(
            [lvl for lvl in self._cluster_levels(pivot_lows) if lvl < close], reverse=True
        )[:n_levels]
        resistance_levels = sorted(
            [lvl for lvl in self._cluster_levels(pivot_highs) if lvl > close]
        )[:n_levels]

        return {
            'support': [round(lvl, 4) for lvl in support_levels],
            'resistance': [round(lvl, 4) for lvl in resistance_levels],
        }

    # ------------------------------------------------------------------ #
    #  Per-bar signal detection helpers                                   #
    # ------------------------------------------------------------------ #

    def _detect_ma_cross_signals(self, current: pd.Series, prev: pd.Series) -> list[dict]:
        signals = []
        for fast, slow in MA_FAST_SLOW_PAIRS:
            cf, cs = f'SMA_{fast}', f'SMA_{slow}'
            cur_f = _safe_float(current.get(cf))
            cur_s = _safe_float(current.get(cs))
            pre_f = _safe_float(prev.get(cf))
            pre_s = _safe_float(prev.get(cs))
            if None in (cur_f, cur_s, pre_f, pre_s):
                continue
            if pre_f <= pre_s and cur_f > cur_s:
                label = 'GOLDEN CROSS' if (fast, slow) == (50, 200) else f'{fast}/{slow} MA BULL CROSS'
                strength = 'STRONG BULLISH' if (fast, slow) == (50, 200) else 'BULLISH'
                signals.append({
                    'signal': label,
                    'description': f'{fast} SMA crossed above {slow} SMA',
                    'strength': strength,
                    'category': 'MA_CROSS',
                    'value': cur_f,
                })
            elif pre_f >= pre_s and cur_f < cur_s:
                label = 'DEATH CROSS' if (fast, slow) == (50, 200) else f'{fast}/{slow} MA BEAR CROSS'
                strength = 'STRONG BEARISH' if (fast, slow) == (50, 200) else 'BEARISH'
                signals.append({
                    'signal': label,
                    'description': f'{fast} SMA crossed below {slow} SMA',
                    'strength': strength,
                    'category': 'MA_CROSS',
                    'value': cur_f,
                })
        return signals

    def _detect_rsi_signals(self, current: pd.Series, prev: pd.Series) -> list[dict]:
        signals = []
        for p in RSI_PERIODS:
            col = f'RSI_{p}'
            rsi = _safe_float(current.get(col))
            prev_rsi = _safe_float(prev.get(col))
            if rsi is None:
                continue
            for os_lvl, ob_lvl in RSI_OS_OB_LEVELS:
                if rsi < os_lvl:
                    signals.append({
                        'signal': f'RSI{p} OVERSOLD (<{os_lvl})',
                        'description': f'RSI({p}): {rsi:.1f}',
                        'strength': 'BULLISH',
                        'category': 'RSI',
                        'value': rsi,
                    })
                elif rsi > ob_lvl:
                    signals.append({
                        'signal': f'RSI{p} OVERBOUGHT (>{ob_lvl})',
                        'description': f'RSI({p}): {rsi:.1f}',
                        'strength': 'BEARISH',
                        'category': 'RSI',
                        'value': rsi,
                    })
            # Midline cross (50) — once per period
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
        return signals

    def _detect_macd_signals(self, current: pd.Series, prev: pd.Series) -> list[dict]:
        signals = []
        for fast, slow, sig in MACD_PARAMS:
            tag = f'_{fast}_{slow}_{sig}'
            macd = _safe_float(current.get(f'MACD{tag}'))
            macd_sig = _safe_float(current.get(f'MACD_Signal{tag}'))
            macd_hist = _safe_float(current.get(f'MACD_Hist{tag}'))
            prev_macd = _safe_float(prev.get(f'MACD{tag}'))
            prev_macd_sig = _safe_float(prev.get(f'MACD_Signal{tag}'))
            prev_macd_hist = _safe_float(prev.get(f'MACD_Hist{tag}'))
            if None in (macd, macd_sig, prev_macd, prev_macd_sig):
                continue

            # Signal line cross
            if prev_macd <= prev_macd_sig and macd > macd_sig:
                signals.append({
                    'signal': f'MACD({fast},{slow},{sig}) BULL CROSS',
                    'description': f'MACD crossed above signal line',
                    'strength': 'STRONG BULLISH',
                    'category': 'MACD',
                    'value': macd,
                })
            elif prev_macd >= prev_macd_sig and macd < macd_sig:
                signals.append({
                    'signal': f'MACD({fast},{slow},{sig}) BEAR CROSS',
                    'description': f'MACD crossed below signal line',
                    'strength': 'STRONG BEARISH',
                    'category': 'MACD',
                    'value': macd,
                })

            # Zero-line cross
            if prev_macd <= 0 < macd:
                signals.append({
                    'signal': f'MACD({fast},{slow},{sig}) ZERO BULL CROSS',
                    'description': 'MACD crossed above zero',
                    'strength': 'BULLISH',
                    'category': 'MACD',
                    'value': macd,
                })
            elif prev_macd >= 0 > macd:
                signals.append({
                    'signal': f'MACD({fast},{slow},{sig}) ZERO BEAR CROSS',
                    'description': 'MACD crossed below zero',
                    'strength': 'BEARISH',
                    'category': 'MACD',
                    'value': macd,
                })

            # Histogram zero cross
            if macd_hist is not None and prev_macd_hist is not None:
                if prev_macd_hist <= 0 < macd_hist:
                    signals.append({
                        'signal': f'MACD({fast},{slow},{sig}) HIST BULL',
                        'description': 'MACD histogram turned positive',
                        'strength': 'BULLISH',
                        'category': 'MACD',
                        'value': macd_hist,
                    })
                elif prev_macd_hist >= 0 > macd_hist:
                    signals.append({
                        'signal': f'MACD({fast},{slow},{sig}) HIST BEAR',
                        'description': 'MACD histogram turned negative',
                        'strength': 'BEARISH',
                        'category': 'MACD',
                        'value': macd_hist,
                    })
        return signals

    def _detect_bb_signals(self, current: pd.Series, prev: pd.Series) -> list[dict]:
        signals = []
        close = _safe_float(current.get('Close'))
        prev_close = _safe_float(prev.get('Close'))
        if close is None:
            return signals

        for p in BB_PERIODS:
            for sd in BB_STD_DEVS:
                sd_tag = str(sd).replace('.', '_')
                upper = _safe_float(current.get(f'BB_{p}_{sd_tag}_Upper'))
                lower = _safe_float(current.get(f'BB_{p}_{sd_tag}_Lower'))
                pct_b = _safe_float(current.get(f'BB_{p}_{sd_tag}_Pct'))
                prev_upper = _safe_float(prev.get(f'BB_{p}_{sd_tag}_Upper'))
                prev_lower = _safe_float(prev.get(f'BB_{p}_{sd_tag}_Lower'))
                if None in (upper, lower):
                    continue

                label = f'BB({p},{sd})'

                if close > upper:
                    signals.append({
                        'signal': f'ABOVE UPPER {label}',
                        'description': f'Price {close:.2f} above upper band {upper:.2f}',
                        'strength': 'EXTREME BULLISH',
                        'category': 'BB_BREAKOUT',
                        'value': close - upper,
                    })
                elif close < lower:
                    signals.append({
                        'signal': f'BELOW LOWER {label}',
                        'description': f'Price {close:.2f} below lower band {lower:.2f}',
                        'strength': 'EXTREME BEARISH',
                        'category': 'BB_BREAKOUT',
                        'value': lower - close,
                    })

                # %B cross 0 / 1
                if pct_b is not None:
                    if pct_b > 1.0:
                        signals.append({
                            'signal': f'{label} %B > 1',
                            'description': f'%B={pct_b:.2f} (overbought)',
                            'strength': 'BEARISH',
                            'category': 'BB_BREAKOUT',
                            'value': pct_b,
                        })
                    elif pct_b < 0.0:
                        signals.append({
                            'signal': f'{label} %B < 0',
                            'description': f'%B={pct_b:.2f} (oversold)',
                            'strength': 'BULLISH',
                            'category': 'BB_BREAKOUT',
                            'value': pct_b,
                        })

                # Band ride: 2 consecutive closes above upper
                if (prev_close is not None and prev_upper is not None
                        and prev_close > prev_upper and close > upper):
                    signals.append({
                        'signal': f'{label} RIDING UPPER BAND',
                        'description': '2 consecutive closes above upper band (strong uptrend)',
                        'strength': 'STRONG BULLISH',
                        'category': 'BB_BREAKOUT',
                        'value': close,
                    })
        return signals

    def _detect_volume_signals(self, current: pd.Series, prev: pd.Series, df_context: pd.DataFrame) -> list[dict]:
        signals = []
        vol = _safe_float(current.get('Volume'))
        if vol is None:
            return signals

        for vm in VOLUME_MAS:
            vol_ma = _safe_float(current.get(f'Volume_MA_{vm}'))
            if vol_ma is None or vol_ma == 0:
                continue
            ratio = vol / vol_ma

            for thresh in VOLUME_SPIKE_THRESHOLDS:
                if ratio > thresh:
                    signals.append({
                        'signal': f'VOLUME SPIKE >{thresh}x (MA{vm})',
                        'description': f'Volume {ratio:.1f}x the {vm}-bar average',
                        'strength': 'SIGNIFICANT',
                        'category': 'VOLUME',
                        'value': ratio,
                    })
            for thresh in VOLUME_LOW_THRESHOLDS:
                if ratio < thresh:
                    signals.append({
                        'signal': f'LOW VOLUME <{thresh}x (MA{vm})',
                        'description': f'Volume {ratio:.1f}x the {vm}-bar average (weak conviction)',
                        'strength': 'NEUTRAL',
                        'category': 'VOLUME',
                        'value': ratio,
                    })

        # Volume divergence over 10 bars
        if len(df_context) >= 10:
            price_chg = (_safe_value(df_context['Close'].iloc[-1])
                         - _safe_value(df_context['Close'].iloc[-10]))
            vol_chg = (_safe_value(df_context['Volume'].iloc[-1])
                       - _safe_value(df_context['Volume'].iloc[-10]))
            if price_chg > 0 and vol_chg < 0:
                signals.append({
                    'signal': 'VOLUME BEARISH DIVERGENCE (10b)',
                    'description': 'Price rising but volume falling over 10 bars',
                    'strength': 'BEARISH',
                    'category': 'VOLUME',
                    'value': price_chg,
                })
            elif price_chg < 0 and vol_chg > 0:
                signals.append({
                    'signal': 'VOLUME BULLISH DIVERGENCE (10b)',
                    'description': 'Price falling but volume rising over 10 bars',
                    'strength': 'BULLISH',
                    'category': 'VOLUME',
                    'value': abs(price_chg),
                })
        return signals

    def _detect_price_action_signals(self, current: pd.Series) -> list[dict]:
        signals = []
        for lb in PRICE_ACTION_LOOKBACKS:
            pc = _safe_float(current.get(f'Price_Change_{lb}'))
            if pc is None:
                continue
            for thresh in PRICE_ACTION_THRESHOLDS:
                if pc > thresh:
                    signals.append({
                        'signal': f'GAIN >{thresh}% ({lb}b)',
                        'description': f'+{pc:.1f}% over {lb} bar(s)',
                        'strength': 'STRONG BULLISH' if pc > thresh * 1.5 else 'BULLISH',
                        'category': 'PRICE_ACTION',
                        'value': pc,
                    })
                elif pc < -thresh:
                    signals.append({
                        'signal': f'LOSS <-{thresh}% ({lb}b)',
                        'description': f'{pc:.1f}% over {lb} bar(s)',
                        'strength': 'STRONG BEARISH' if pc < -thresh * 1.5 else 'BEARISH',
                        'category': 'PRICE_ACTION',
                        'value': pc,
                    })
        return signals

    def _detect_hl_signals(self, current: pd.Series) -> list[dict]:
        signals = []
        close = _safe_float(current.get('Close'))
        if close is None:
            return signals

        for lb in HL_LOOKBACKS:
            high_col = f'High_{lb}b'
            low_col = f'Low_{lb}b'
            high_val = _safe_float(current.get(high_col))
            low_val = _safe_float(current.get(low_col))
            if high_val is None or low_val is None:
                continue

            for prox in HL_PROXIMITIES:
                if high_val != 0 and close >= high_val * (1 - prox):
                    signals.append({
                        'signal': f'WITHIN {int(prox*100)}% OF {lb}b HIGH',
                        'description': f'Close {close:.2f} within {prox*100:.0f}% of {lb}-bar high {high_val:.2f}',
                        'strength': 'EXTREME BULLISH' if prox <= 0.01 else 'BULLISH',
                        'category': 'RANGE',
                        'value': close,
                    })
                if low_val != 0 and close <= low_val * (1 + prox):
                    signals.append({
                        'signal': f'WITHIN {int(prox*100)}% OF {lb}b LOW',
                        'description': f'Close {close:.2f} within {prox*100:.0f}% of {lb}-bar low {low_val:.2f}',
                        'strength': 'EXTREME BEARISH' if prox <= 0.01 else 'BEARISH',
                        'category': 'RANGE',
                        'value': close,
                    })
        return signals

    def _detect_ma_distance_signals(self, current: pd.Series) -> list[dict]:
        signals = []
        for p in MA_DIST_PERIODS:
            dist_col = f'Dist_SMA_{p}'
            dist = _safe_float(current.get(dist_col))
            if dist is None:
                continue
            for thresh in MA_DIST_THRESHOLDS:
                if dist > thresh:
                    signals.append({
                        'signal': f'>{thresh}% ABOVE {p}SMA',
                        'description': f'{dist:.1f}% above {p}-period SMA',
                        'strength': 'BEARISH',
                        'category': 'MA_DISTANCE',
                        'value': dist,
                    })
                elif dist < -thresh:
                    signals.append({
                        'signal': f'>{thresh}% BELOW {p}SMA',
                        'description': f'{abs(dist):.1f}% below {p}-period SMA',
                        'strength': 'BULLISH',
                        'category': 'MA_DISTANCE',
                        'value': dist,
                    })
        return signals

    def _detect_sr_signals(
        self, current: pd.Series, sr_zones: dict, window: int, proximity: float
    ) -> list[dict]:
        """Generate signals when price is near a support or resistance level."""
        signals = []
        close = _safe_value(current['Close'])
        if close == 0:
            return signals

        for lvl in sr_zones.get('support', []):
            if lvl == 0:
                continue
            dist_pct = abs(close - lvl) / lvl
            if dist_pct <= proximity:
                signals.append({
                    'signal': f'NEAR SUPPORT (w={window}, prox={int(proximity*100)}%)',
                    'description': f'Price ${close:.2f} near support ${lvl:.2f} ({dist_pct*100:.1f}% away)',
                    'strength': 'BULLISH',
                    'category': 'SUPPORT_RESISTANCE',
                    'value': lvl,
                })

        for lvl in sr_zones.get('resistance', []):
            if lvl == 0:
                continue
            dist_pct = abs(close - lvl) / lvl
            if dist_pct <= proximity:
                signals.append({
                    'signal': f'NEAR RESISTANCE (w={window}, prox={int(proximity*100)}%)',
                    'description': f'Price ${close:.2f} near resistance ${lvl:.2f} ({dist_pct*100:.1f}% away)',
                    'strength': 'BEARISH',
                    'category': 'SUPPORT_RESISTANCE',
                    'value': lvl,
                })

        return signals

    def _detect_stochastic_signals(self, current: pd.Series, prev: pd.Series) -> list[dict]:
        signals = []
        if 'Stoch_K' not in current.index or pd.isna(current.get('Stoch_K')):
            return signals

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
        return signals

    def _detect_adx_signals(self, current: pd.Series) -> list[dict]:
        signals = []
        adx = _safe_float(current.get('ADX'))
        if adx is None:
            return signals

        plus_di = _safe_value(current.get('Plus_DI', 0))
        minus_di = _safe_value(current.get('Minus_DI', 0))
        direction = 'UP' if plus_di > minus_di else 'DOWN'

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
        return signals

    def _detect_ichimoku_signals(self, current: pd.Series, prev: pd.Series) -> list[dict]:
        signals = []
        cols = ['Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_SpanA', 'Ichimoku_SpanB']
        if any(pd.isna(current.get(col, float('nan'))) for col in cols):
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

        if prev_tenkan <= prev_kijun and tenkan > kijun:
            signals.append({
                'signal': 'ICHIMOKU TK BULL CROSS',
                'description': f'Tenkan ({tenkan:.2f}) crossed above Kijun ({kijun:.2f})',
                'strength': 'STRONG BULLISH',
                'category': 'ICHIMOKU',
                'value': tenkan,
            })
        elif prev_tenkan >= prev_kijun and tenkan < kijun:
            signals.append({
                'signal': 'ICHIMOKU TK BEAR CROSS',
                'description': f'Tenkan ({tenkan:.2f}) crossed below Kijun ({kijun:.2f})',
                'strength': 'STRONG BEARISH',
                'category': 'ICHIMOKU',
                'value': tenkan,
            })

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

    def _detect_obv_cmf_signals(self, current: pd.Series, prev: pd.Series, df_context: pd.DataFrame) -> list[dict]:
        signals = []

        if len(df_context) >= 20 and 'OBV' in df_context.columns:
            recent = df_context.tail(20)
            price_change = (_safe_value(recent['Close'].iloc[-1])
                            - _safe_value(recent['Close'].iloc[0]))
            obv_change = (_safe_value(recent['OBV'].iloc[-1])
                          - _safe_value(recent['OBV'].iloc[0]))

            if price_change > 0 and obv_change < 0:
                signals.append({
                    'signal': 'OBV BEARISH DIVERGENCE',
                    'description': 'Price rising but OBV falling',
                    'strength': 'BEARISH',
                    'category': 'OBV_CMF',
                    'value': _safe_value(current.get('OBV')),
                })
            elif price_change < 0 and obv_change > 0:
                signals.append({
                    'signal': 'OBV BULLISH DIVERGENCE',
                    'description': 'Price falling but OBV rising (accumulation)',
                    'strength': 'STRONG BULLISH',
                    'category': 'OBV_CMF',
                    'value': _safe_value(current.get('OBV')),
                })

        obv_now = _safe_float(current.get('OBV'))
        obv_ema_now = _safe_float(current.get('OBV_EMA'))
        prev_obv = _safe_float(prev.get('OBV', obv_now))
        prev_obv_ema = _safe_float(prev.get('OBV_EMA', obv_ema_now))

        if None not in (obv_now, obv_ema_now, prev_obv, prev_obv_ema):
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

        cmf = _safe_float(current.get('CMF'))
        prev_cmf = _safe_float(prev.get('CMF', cmf))
        if cmf is not None:
            if cmf > 0.1:
                signals.append({
                    'signal': 'CMF STRONG BUYING PRESSURE',
                    'description': f'Chaikin Money Flow: {cmf:.3f} (accumulation)',
                    'strength': 'BULLISH',
                    'category': 'OBV_CMF',
                    'value': cmf,
                })
            elif cmf < -0.1:
                signals.append({
                    'signal': 'CMF STRONG SELLING PRESSURE',
                    'description': f'Chaikin Money Flow: {cmf:.3f} (distribution)',
                    'strength': 'BEARISH',
                    'category': 'OBV_CMF',
                    'value': cmf,
                })
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

    # ------------------------------------------------------------------ #
    #  Bar-level aggregator                                               #
    # ------------------------------------------------------------------ #

    def _detect_signals_at_bar(
        self, df: pd.DataFrame, i: int, sr_cache: dict[int, dict]
    ) -> list[dict]:
        """Run all detectors for bar index i; returns list of signal dicts."""
        current = df.iloc[i]
        prev = df.iloc[i - 1] if i > 0 else current
        df_context = df.iloc[max(0, i - 49): i + 1]

        signals: list[dict] = []
        signals.extend(self._detect_ma_cross_signals(current, prev))
        signals.extend(self._detect_rsi_signals(current, prev))
        signals.extend(self._detect_macd_signals(current, prev))
        signals.extend(self._detect_bb_signals(current, prev))
        signals.extend(self._detect_volume_signals(current, prev, df_context))
        signals.extend(self._detect_price_action_signals(current))
        signals.extend(self._detect_hl_signals(current))
        signals.extend(self._detect_ma_distance_signals(current))
        signals.extend(self._detect_stochastic_signals(current, prev))
        signals.extend(self._detect_adx_signals(current))
        signals.extend(self._detect_ichimoku_signals(current, prev))
        signals.extend(self._detect_obv_cmf_signals(current, prev, df_context))

        # S/R — multiple window × proximity combos
        for win in SR_PIVOT_WINDOWS:
            if win not in sr_cache:
                sr_cache[win] = self._find_support_resistance(df.iloc[:i + 1], window=win)
            for prox in SR_PROXIMITIES:
                signals.extend(self._detect_sr_signals(current, sr_cache[win], win, prox))

        # Attach timestamp
        ts = df.index[i]
        ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
        for s in signals:
            s['timestamp'] = ts_str

        return signals

    # ------------------------------------------------------------------ #
    #  Main signal detection (all bars)                                   #
    # ------------------------------------------------------------------ #

    def detect_signals(self) -> list[dict]:
        """Detect signals and compute confluence across every bar in self.data."""
        print("Detecting signals across all bars...")
        df = self.data.copy()
        min_warmup = max(
            200,  # longest SMA
            max(s for _, s, _ in MACD_PARAMS),
            max(HL_LOOKBACKS),
        )
        start_bar = min(min_warmup, len(df) - 1)

        all_signals: list[dict] = []
        bar_confluence: list[dict] = []
        sr_cache: dict[int, dict] = {}

        for i in range(start_bar, len(df)):
            if i % 20 == 0:
                sr_cache.clear()
            bar_signals = self._detect_signals_at_bar(df, i, sr_cache)
            all_signals.extend(bar_signals)

            ts_raw = df.index[i]
            ts_str = ts_raw.isoformat() if isinstance(ts_raw, pd.Timestamp) else str(ts_raw)
            rank = ConfluenceRanker.rank_signals(bar_signals)
            rank['timestamp'] = ts_str
            bar_confluence.append(rank)

        self.historical_signals = all_signals
        self.bar_confluence = bar_confluence

        if len(df) > 0:
            self.current = df.iloc[-1]
            last_ts = df.index[-1]
            last_ts_str = last_ts.isoformat() if hasattr(last_ts, 'isoformat') else str(last_ts)
            self.signals = [s for s in all_signals if s.get('timestamp') == last_ts_str]
            self._sr_zones = self._find_support_resistance(df)
            self._latest_confluence = bar_confluence[-1] if bar_confluence else {}

        print(f"Detected {len(all_signals)} signals across {len(df) - start_bar} bars "
              f"({len(self.signals)} on latest bar)")
        return self.signals

    # ------------------------------------------------------------------ #
    #  Export                                                              #
    # ------------------------------------------------------------------ #

    def export_json(self, now: datetime | None = None) -> Path:
        """Export signals and data to JSON."""
        print("Exporting to JSON...")
        now = now or datetime.now()
        current = self.current
        sr = getattr(self, '_sr_zones', {'support': [], 'resistance': []})

        bullish = sum(1 for s in self.signals if 'BULLISH' in s['strength'])
        bearish = sum(1 for s in self.signals if 'BEARISH' in s['strength'])
        neutral = sum(1 for s in self.signals if 'NEUTRAL' in s['strength'])
        other = len(self.signals) - bullish - bearish - neutral

        by_category: dict[str, int] = {}
        for sig in self.signals:
            cat = sig['category']
            by_category[cat] = by_category.get(cat, 0) + 1

        data = {
            'metadata': {
                'symbol': self.symbol,
                'timestamp': now.isoformat(),
                'data_period': self.period,
                'interval': self.interval,
                'total_bars': len(self.data),
                'signals_on_latest_bar': len(self.signals),
                'total_historical_signals': len(self.historical_signals),
            },
            'current_price': {
                'close': _safe_float(current['Close']),
                'open': _safe_float(current['Open']),
                'high': _safe_float(current['High']),
                'low': _safe_float(current['Low']),
                'volume': int(current['Volume']) if not pd.isna(current['Volume']) else 0,
                'change_pct_1b': _safe_float(current.get('Price_Change_1')),
                'change_pct_5b': _safe_float(current.get('Price_Change_5')),
                'change_pct_10b': _safe_float(current.get('Price_Change_10')),
                'change_pct_20b': _safe_float(current.get('Price_Change_20')),
            },
            'indicators': {
                'RSI_14': _safe_float(current.get('RSI_14')),
                'MACD': _safe_float(current.get('MACD')),
                'MACD_Signal': _safe_float(current.get('MACD_Signal')),
                'MACD_Hist': _safe_float(current.get('MACD_Hist')),
                'ADX': _safe_float(current.get('ADX')),
                'Stochastic_K': _safe_float(current.get('Stoch_K')),
                'Stochastic_D': _safe_float(current.get('Stoch_D')),
                'ATR': _safe_float(current.get('ATR')),
                'BB_20_2sd_Position': _safe_float(current.get('BB_20_2_0_Pct')),
                'Volume_Ratio_MA20': _safe_float(
                    current['Volume'] / current.get('Volume_MA_20', current['Volume'])
                    if _safe_float(current.get('Volume_MA_20')) else None
                ),
                'OBV': _safe_float(current.get('OBV')),
                'CMF': _safe_float(current.get('CMF')),
                'Ichimoku_Tenkan': _safe_float(current.get('Ichimoku_Tenkan')),
                'Ichimoku_Kijun': _safe_float(current.get('Ichimoku_Kijun')),
                'Ichimoku_SpanA': _safe_float(current.get('Ichimoku_SpanA')),
                'Ichimoku_SpanB': _safe_float(current.get('Ichimoku_SpanB')),
            },
            'moving_averages': (
                {f'SMA_{p}': _safe_float(current.get(f'SMA_{p}'))
                 for p in sorted({p for pair in MA_FAST_SLOW_PAIRS for p in pair} | {50, 100, 200})}
                | {f'EMA_{p}': _safe_float(current.get(f'EMA_{p}'))
                   for p in [10, 20, 50, 100, 200]}
            ),
            'support_resistance': sr,
            'signals': self.signals,
            'signal_summary': {
                'total': len(self.signals),
                'bullish': bullish,
                'bearish': bearish,
                'neutral': neutral,
                'other': other,
                'by_category': by_category,
            },
            'confluence': getattr(self, '_latest_confluence', {}),
            'historical_signal_count': len(self.historical_signals),
        }

        filename = self.output_dir / f"{self.symbol}_{now.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, cls=SafeJSONEncoder)

        print(f"JSON saved: {filename}")
        return filename

    def export_markdown(self, now: datetime | None = None) -> Path:
        """Export signals to formatted Markdown report."""
        print("Exporting to Markdown...")
        now = now or datetime.now()
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
        neutral = sum(1 for s in self.signals if 'NEUTRAL' in s['strength'])
        other = len(self.signals) - bullish - bearish - neutral

        by_category: dict[str, int] = {}
        for sig in self.signals:
            by_category[sig['category']] = by_category.get(sig['category'], 0) + 1

        signals_by_cat: dict[str, list] = {}
        for sig in self.signals:
            signals_by_cat.setdefault(sig['category'], []).append(sig)

        sr_support_str = ' | '.join(f"${l:.2f}" for l in sr['support']) or 'None detected'
        sr_resist_str = ' | '.join(f"${l:.2f}" for l in sr['resistance']) or 'None detected'

        md = f"""# Technical Analysis Report: {self.symbol}

**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S')}
**Period:** {self.period}  |  **Interval:** {self.interval}
**Latest-bar Signals:** {len(self.signals)}  |  **Total Historical Signals:** {len(self.historical_signals)}

---

## Current Price Data

| Metric | Value |
|--------|-------|
| **Close** | ${safe_val(current['Close'])} |
| **Open** | ${safe_val(current['Open'])} |
| **High** | ${safe_val(current['High'])} |
| **Low** | ${safe_val(current['Low'])} |
| **Volume** | {int(current['Volume']) if not pd.isna(current['Volume']) else 0:,} |
| **Change 1b %** | {safe_val(current.get('Price_Change_1', 0))}% |
| **Change 5b %** | {safe_val(current.get('Price_Change_5', 0))}% |
| **Change 10b %** | {safe_val(current.get('Price_Change_10', 0))}% |
| **Change 20b %** | {safe_val(current.get('Price_Change_20', 0))}% |

---

## Technical Indicators

| Indicator | Value |
|-----------|-------|
| **RSI(14)** | {safe_val(current.get('RSI_14'))} |
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
| **100** | {safe_val(current.get('SMA_100'))} | {safe_val(current.get('EMA_100'))} |
| **200** | {safe_val(current.get('SMA_200'))} | {safe_val(current.get('EMA_200'))} |

---

## Support & Resistance Zones

| Type | Levels |
|------|--------|
| **Support** | {sr_support_str} |
| **Resistance** | {sr_resist_str} |

---

## Detected Signals — Latest Bar ({len(self.signals)} Total)

| Strength | Count |
|----------|-------|
| Bullish | {bullish} |
| Bearish | {bearish} |
| Neutral | {neutral} |
| Other (Trending/Significant) | {other} |

### Signals by Category

"""
        for cat, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
            md += f"- **{cat}**: {count} signal{'s' if count > 1 else ''}\n"

        md += "\n---\n\n### All Latest-Bar Signals\n"

        for cat in sorted(signals_by_cat.keys()):
            md += f"\n#### {cat.replace('_', ' ').title()}\n\n"
            for sig in signals_by_cat[cat]:
                strength = sig['strength']
                if 'BULLISH' in strength:
                    emoji = "🟢"
                elif 'BEARISH' in strength:
                    emoji = "🔴"
                else:
                    emoji = "⚪"
                md += f"**{emoji} {sig['signal']}**\n"
                md += f"- {sig['description']}\n"
                md += f"- Strength: {strength}\n"
                md += f"- Value: {safe_val(sig['value'])}\n\n"

        # Confluence section
        conf = getattr(self, '_latest_confluence', {})
        if conf:
            sign = '+' if conf['net_score'] >= 0 else ''
            md += f"""
---

## Confluence Ranking — Latest Bar

| Metric | Value |
|--------|-------|
| **Net Score** | {sign}{conf['net_score']} |
| **Bias** | {conf['bias']} |
| **Confidence** | {conf['confidence']} |
| **Bullish Signals** | {conf['bullish_signal_count']} (score {conf['bullish_score']}) |
| **Bearish Signals** | {conf['bearish_signal_count']} (score {conf['bearish_score']}) |
| **Agreement Ratio** | {conf['agreement_ratio']:.1%} |

> *{ConfluenceRanker.interpret(conf)}*

"""

        md += f"\n---\n\n*Total historical signals (all bars): {len(self.historical_signals)}*\n"
        md += "*Report generated by YFinance Signal Detector — April 500 Edition*\n"

        filename = self.output_dir / f"{self.symbol}_{now.strftime('%Y%m%d_%H%M%S')}.md"
        with open(filename, 'w') as f:
            f.write(md)

        print(f"Markdown saved: {filename}")
        return filename

    def export_historical_json(self, now: datetime | None = None) -> Path:
        """Export full historical signals and per-bar confluence to JSON."""
        print("Exporting historical signals to JSON...")
        now = now or datetime.now()
        filename = (self.output_dir
                    / f"{self.symbol}_{now.strftime('%Y%m%d_%H%M%S')}_historical.json")
        with open(filename, 'w') as f:
            json.dump(
                {
                    'symbol': self.symbol,
                    'signals': self.historical_signals,
                    'bar_confluence': self.bar_confluence,
                },
                f, indent=2, cls=SafeJSONEncoder
            )
        print(f"Historical JSON saved: {filename}")
        return filename

    def run_complete_analysis(self) -> dict:
        """Run complete pipeline: fetch → calculate → detect → export."""
        print(f"\n{'='*60}")
        print(f"SIGNAL ANALYSIS: {self.symbol}  [{self.period} / {self.interval}]")
        print(f"{'='*60}\n")

        self.fetch_data()
        self.calculate_indicators()
        self.detect_signals()

        now = datetime.now()
        json_file = self.export_json(now=now)
        md_file = self.export_markdown(now=now)
        hist_file = self.export_historical_json(now=now)

        conf = getattr(self, '_latest_confluence', {})
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"JSON:        {json_file}")
        print(f"MD:          {md_file}")
        print(f"Historical:  {hist_file}")
        print(f"Latest-bar signals:   {len(self.signals)}")
        print(f"Total historical:     {len(self.historical_signals)}")
        if conf:
            sign = '+' if conf['net_score'] >= 0 else ''
            print(f"Confluence:   {conf['bias']} {sign}{conf['net_score']} ({conf['confidence']})")
        print(f"{'='*60}\n")

        return {
            'json_file': json_file,
            'md_file': md_file,
            'hist_file': hist_file,
            'signals': self.signals,
            'historical_signals': self.historical_signals,
            'confluence': conf,
            'bar_confluence': self.bar_confluence,
        }


def run_multi_timeframe(
    symbol: str,
    output_dir: str = 'signal_reports',
    configs: list[dict] | None = None,
) -> dict:
    """Run SignalDetectorExporter across multiple timeframes and return a weighted
    composite confluence score plus a per-timeframe breakdown.

    Args:
        symbol: Ticker symbol (e.g. 'AAPL').
        output_dir: Directory for report files.
        configs: List of timeframe config dicts with keys label, period, interval,
                 weight, use_case. Defaults to TIMEFRAME_CONFIGS.

    Returns dict with keys:
        timeframes: list of per-timeframe result dicts
        composite_score: weighted net score across all timeframes
        composite_bias: BULLISH / BEARISH / NEUTRAL
        composite_confidence: HIGH / MEDIUM / LOW
        symbol: the ticker
    """
    configs = configs or TIMEFRAME_CONFIGS
    print(f"\n{'='*60}")
    print(f"MULTI-TIMEFRAME ANALYSIS: {symbol}")
    print(f"{'='*60}")

    results = []
    total_weight = 0.0
    weighted_score = 0.0

    for cfg in configs:
        label = cfg['label']
        period = cfg['period']
        interval = cfg['interval']
        weight = cfg['weight']
        use_case = cfg['use_case']
        print(f"\n  [{label}] period={period} interval={interval} weight={weight}")
        try:
            det = SignalDetectorExporter(symbol=symbol, period=period, output_dir=output_dir)
            # Override interval from config (not just from PERIOD_INTERVAL_MAP)
            det.interval = interval
            det.fetch_data()
            det.calculate_indicators()
            det.detect_signals()

            conf = getattr(det, '_latest_confluence', {})
            net = conf.get('net_score', 0.0)
            weighted_score += net * weight
            total_weight += weight

            results.append({
                'label': label,
                'period': period,
                'interval': interval,
                'weight': weight,
                'use_case': use_case,
                'bias': conf.get('bias', 'N/A'),
                'net_score': net,
                'confidence': conf.get('confidence', 'N/A'),
                'bullish_score': conf.get('bullish_score', 0),
                'bearish_score': conf.get('bearish_score', 0),
                'signal_count': conf.get('signal_count', 0),
                'agreement_ratio': conf.get('agreement_ratio', 0.5),
                'latest_bar_signal_count': len(det.signals),
            })
            print(f"     bias={conf.get('bias')}  net={net:+.2f}  confidence={conf.get('confidence')}")
        except Exception as e:
            print(f"     ERROR: {e}")
            results.append({
                'label': label, 'period': period, 'interval': interval,
                'weight': weight, 'use_case': use_case,
                'bias': 'ERROR', 'net_score': 0.0, 'confidence': 'N/A',
                'error': str(e),
            })

    composite = weighted_score / total_weight if total_weight > 0 else 0.0
    composite_bias = 'BULLISH' if composite > 0.5 else ('BEARISH' if composite < -0.5 else 'NEUTRAL')
    composite_confidence = 'HIGH' if abs(composite) >= 3.0 else ('MEDIUM' if abs(composite) >= 1.0 else 'LOW')

    print(f"\n{'='*60}")
    print(f"COMPOSITE: {composite_bias}  score={composite:+.2f}  confidence={composite_confidence}")
    print(f"{'='*60}\n")

    return {
        'symbol': symbol,
        'timeframes': results,
        'composite_score': round(composite, 2),
        'composite_bias': composite_bias,
        'composite_confidence': composite_confidence,
    }


def export_multi_timeframe_markdown(mtf: dict, output_dir: str = 'signal_reports') -> Path:
    """Write a multi-timeframe outlook markdown report."""
    symbol = mtf['symbol']
    composite = mtf['composite_score']
    sign = '+' if composite >= 0 else ''
    bias_emoji = '🟢' if mtf['composite_bias'] == 'BULLISH' else ('🔴' if mtf['composite_bias'] == 'BEARISH' else '⚪')

    lines = [
        f"# Multi-Timeframe Outlook: {symbol}",
        f"",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Composite Score:** {sign}{composite}  |  "
        f"**Bias:** {bias_emoji} {mtf['composite_bias']}  |  "
        f"**Confidence:** {mtf['composite_confidence']}",
        f"",
        f"---",
        f"",
        f"## Timeframe Breakdown",
        f"",
        f"| Timeframe | Bias | Net Score | Confidence | Agree% | Signals | Use Case |",
        f"|-----------|------|-----------|------------|--------|---------|----------|",
    ]
    for r in mtf['timeframes']:
        if 'error' in r:
            lines.append(
                f"| {r['label']} | ERROR | N/A | N/A | N/A | N/A | {r['use_case']} |"
            )
            continue
        b_emoji = '🟢' if r['bias'] == 'BULLISH' else ('🔴' if r['bias'] == 'BEARISH' else '⚪')
        s = r['net_score']
        lines.append(
            f"| **{r['label']}** | {b_emoji} {r['bias']} | {'+' if s >= 0 else ''}{s:.2f} "
            f"| {r['confidence']} | {r['agreement_ratio']:.0%} "
            f"| {r['signal_count']} | {r['use_case']} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## Composite Weighted Outlook",
        f"",
        f"**Score:** {sign}{composite}  "
        f"**Bias:** {bias_emoji} {mtf['composite_bias']}  "
        f"**Confidence:** {mtf['composite_confidence']}",
        f"",
        f"> Weights: " + ', '.join(
            f"{r['label']}×{r['weight']}" for r in mtf['timeframes']
        ),
        f"",
        f"---",
        f"",
        f"*Report generated by YFinance Signal Detector — April 500 Edition*",
    ]

    path = Path(output_dir)
    path.mkdir(exist_ok=True)
    filename = path / f"{symbol}_mtf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    filename.write_text('\n'.join(lines))
    print(f"Multi-timeframe MD saved: {filename}")
    return filename


# ============ MAIN EXECUTION ============

if __name__ == "__main__":
    print("\n=== SINGLE STOCK ANALYSIS ===\n")
    detector = SignalDetectorExporter(symbol="AAPL", period="1y")
    results = detector.run_complete_analysis()
    print(f"\nLatest-bar signals: {len(results['signals'])}")
    print(f"Total historical:   {len(results['historical_signals'])}")
    print(f"Files saved to:     {detector.output_dir}")

    print("\n\n=== MULTI-PERIOD ANALYSIS ===\n")
    for period in ['6mo', '3mo', '1mo', '5d']:
        try:
            d = SignalDetectorExporter(symbol="AAPL", period=period)
            d.run_complete_analysis()
        except Exception as e:
            print(f"  {period}: {e}")

    print("\n\n=== BATCH ANALYSIS ===\n")
    for sym in ['MSFT', 'GOOGL', 'TSLA']:
        try:
            SignalDetectorExporter(symbol=sym, period="6mo").run_complete_analysis()
        except Exception as e:
            print(f"  Error with {sym}: {e}")

    print("\n\n=== MULTI-TIMEFRAME ANALYSIS ===\n")
    for sym in ['AAPL', 'MSFT', 'TSLA']:
        try:
            mtf = run_multi_timeframe(symbol=sym, output_dir='signal_reports')
            mtf_md = export_multi_timeframe_markdown(mtf, output_dir='signal_reports')
            print(f"  {sym}: {mtf['composite_bias']} score={mtf['composite_score']:+.2f} "
                  f"confidence={mtf['composite_confidence']}  -> {mtf_md.name}")
        except Exception as e:
            print(f"  Error with {sym}: {e}")

    print("\nAll analyses complete!")
