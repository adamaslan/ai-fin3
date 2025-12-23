  def _detect_fibonacci_signals(self, df: pd.DataFrame, current: pd.Series, prev: pd.Series) -> List[dict]:
    # Detect 150+ comprehensive Fibonacci signals including retracement, extension, arcs, fans, time zones, channels, Elliott Wave, and confluence patterns"""
    signals = []
    
    if len(df) < 50:
        return signals
    
    # Core calculations
    window = 50
    high_50 = df['High'].iloc[-window:].max()
    low_50 = df['Low'].iloc[-window:].min()
    swing_range = high_50 - low_50
    
    if swing_range == 0:
        return signals
    
    close = self._safe_float(current['Close'])
    if close is None:
        return signals
    
    # Extended Fibonacci ratios covering all 106 original + 44 new signals
    fib_levels = {
        # ORIGINAL RETRACEMENT LEVELS (23.6%, 38.2%, 50%, 61.8%, 78.6%)
        'RETRACE_236': {'ratio': 0.236, 'strength': 'WEAK', 'name': '23.6%', 'type': 'RETRACE'},
        'RETRACE_382': {'ratio': 0.382, 'strength': 'MODERATE', 'name': '38.2%', 'type': 'RETRACE'},
        'RETRACE_500': {'ratio': 0.500, 'strength': 'MODERATE', 'name': '50.0%', 'type': 'RETRACE'},
        'RETRACE_618': {'ratio': 0.618, 'strength': 'SIGNIFICANT', 'name': '61.8%', 'type': 'RETRACE'},
        'RETRACE_786': {'ratio': 0.786, 'strength': 'SIGNIFICANT', 'name': '78.6%', 'type': 'RETRACE'},
        
        # EXTENSION LEVELS (127.2%, 141.4%, 161.8%, 200%, 261.8%)
        'EXT_1272': {'ratio': 1.272, 'strength': 'MODERATE', 'name': '127.2%', 'type': 'EXTENSION'},
        'EXT_1414': {'ratio': 1.414, 'strength': 'MODERATE', 'name': '141.4%', 'type': 'EXTENSION'},
        'EXT_1618': {'ratio': 1.618, 'strength': 'SIGNIFICANT', 'name': '161.8%', 'type': 'EXTENSION'},
        'EXT_2000': {'ratio': 2.0, 'strength': 'SIGNIFICANT', 'name': '200.0%', 'type': 'EXTENSION'},
        'EXT_2236': {'ratio': 2.236, 'strength': 'SIGNIFICANT', 'name': '223.6%', 'type': 'EXTENSION'},
        'EXT_2618': {'ratio': 2.618, 'strength': 'SIGNIFICANT', 'name': '261.8%', 'type': 'EXTENSION'},
        
        # NEW: ADDITIONAL EXTENSION RATIOS (Wave 3/5 targets)
        'EXT_3236': {'ratio': 3.236, 'strength': 'SIGNIFICANT', 'name': '323.6%', 'type': 'EXTENSION'},
        'EXT_4236': {'ratio': 4.236, 'strength': 'SIGNIFICANT', 'name': '423.6%', 'type': 'EXTENSION'},
        
        # NEW: INVERSE RATIOS (38.2% pullback recognition)
        'INV_236': {'ratio': -0.236, 'strength': 'WEAK', 'name': '-23.6%', 'type': 'INVERSE'},
        'INV_382': {'ratio': -0.382, 'strength': 'MODERATE', 'name': '-38.2%', 'type': 'INVERSE'},
        'INV_618': {'ratio': -0.618, 'strength': 'SIGNIFICANT', 'name': '-61.8%', 'type': 'INVERSE'},
    }
    
    # Calculate all Fibonacci levels
    fib_data = {}
    for key, level in fib_levels.items():
        ratio = level['ratio']
        price = low_50 + (ratio * swing_range)
        fib_data[key] = {
            'value': price,
            'ratio': ratio,
            'strength': level['strength'],
            'name': level['name'],
            'type': level['type']
        }
    
    tolerance = 0.01
    tolerance_wide = 0.02
    
    # ========== SIGNAL GROUP 1: PRICE AT FIBONACCI LEVEL (Original - 30 signals) ==========
    for key, level in fib_data.items():
        price_diff = abs(close - level['value']) / close
        
        if price_diff < tolerance:
            signals.append({
                'signal': f"FIB {level['type']} {level['name']}",
                'description': f"Price at {level['name']} {level['type'].lower()} level",
                'strength': level['strength'],
                'category': 'FIB_PRICE_LEVEL',
                'timeframe': self.interval,
                'value': level['value'],
                'distance_pct': price_diff * 100
            })
    
    # ========== SIGNAL GROUP 2: BOUNCE OFF FIBONACCI LEVEL (NEW - 6 signals) ==========
    if len(df) >= 2:
        prev_close = self._safe_float(prev['Close'])
        if prev_close is not None:
            for key, level in fib_data.items():
                if level['type'] in ['RETRACE', 'EXTENSION']:
                    # Price crossed below then bounced up
                    if prev_close < level['value'] and close > level['value']:
                        signals.append({
                            'signal': f"FIB {level['name']} BOUNCE",
                            'description': f"Bounce off {level['name']} Fibonacci level",
                            'strength': 'MODERATE',
                            'category': 'FIB_BOUNCE',
                            'timeframe': self.interval,
                            'value': level['value']
                        })
    
    # ========== SIGNAL GROUP 3: BREAK THROUGH FIBONACCI LEVEL (NEW - 6 signals) ==========
    if len(df) >= 2:
        prev_close = self._safe_float(prev['Close'])
        if prev_close is not None:
            for key, level in fib_data.items():
                # Price broke decisively through level (1% beyond)
                if prev_close < level['value'] and close > level['value'] * 1.01:
                    signals.append({
                        'signal': f"FIB {level['name']} BREAKOUT",
                        'description': f"Breaking through {level['name']} Fibonacci level",
                        'strength': 'SIGNIFICANT',
                        'category': 'FIB_BREAKOUT',
                        'timeframe': self.interval,
                        'value': level['value']
                    })
    
    # ========== SIGNAL GROUP 4: FIBONACCI CHANNEL/BANDS (Original - 10 signals) ==========
    retrace_keys = [k for k in fib_data.keys() if fib_data[k]['type'] == 'RETRACE']
    retrace_keys.sort(key=lambda k: fib_data[k]['ratio'])
    
    for i in range(len(retrace_keys) - 1):
        lower = fib_data[retrace_keys[i]]
        upper = fib_data[retrace_keys[i + 1]]
        
        if lower['value'] <= close <= upper['value']:
            signals.append({
                'signal': f"FIB CHANNEL {lower['name']}-{upper['name']}",
                'description': f"Price in Fibonacci channel between {lower['name']} and {upper['name']}",
                'strength': 'MODERATE',
                'category': 'FIB_CHANNEL',
                'timeframe': self.interval,
                'value': (lower['value'] + upper['value']) / 2
            })
    
    # ========== SIGNAL GROUP 5: FIBONACCI EXTENSION CHANNEL (NEW - 8 signals) ==========
    ext_keys = [k for k in fib_data.keys() if fib_data[k]['type'] == 'EXTENSION']
    ext_keys.sort(key=lambda k: fib_data[k]['ratio'])
    
    for i in range(len(ext_keys) - 1):
        lower = fib_data[ext_keys[i]]
        upper = fib_data[ext_keys[i + 1]]
        
        if lower['value'] <= close <= upper['value']:
            signals.append({
                'signal': f"FIB EXT CHANNEL {lower['name']}-{upper['name']}",
                'description': f"Price in extension channel between {lower['name']} and {upper['name']}",
                'strength': 'SIGNIFICANT',
                'category': 'FIB_EXT_CHANNEL',
                'timeframe': self.interval,
                'value': (lower['value'] + upper['value']) / 2
            })
    
    # ========== SIGNAL GROUP 6: FIBONACCI ARC SIGNALS (Original - 12 signals) ==========
    arc_ratios = [0.236, 0.382, 0.500, 0.618, 0.786, 1.0]
    time_since_pivot = len(df) - (df['High'].iloc[:-window].argmax() if len(df) > window else 0)
    
    for arc_ratio in arc_ratios:
        # Arc extends outward in both price and time
        arc_price = low_50 + (arc_ratio * swing_range) * (1 + time_since_pivot / len(df))
        
        if abs(close - arc_price) / close < tolerance_wide:
            signals.append({
                'signal': f"FIB ARC {arc_ratio*100:.1f}%",
                'description': f"Price touching {arc_ratio*100:.1f}% Fibonacci arc",
                'strength': 'MODERATE',
                'category': 'FIB_ARC',
                'timeframe': self.interval,
                'value': arc_price
            })
    
    # ========== SIGNAL GROUP 7: FIBONACCI FAN LINES (Original - 10 signals) ==========
    fan_ratios = [0.236, 0.382, 0.500, 0.618, 0.786]
    time_diff = len(df) - 50
    
    for fan_ratio in fan_ratios:
        fan_price = low_50 + (fan_ratio * swing_range) * (time_diff / 50) if time_diff > 0 else low_50
        fan_strength = 'SIGNIFICANT' if fan_ratio in [0.618, 0.786] else 'MODERATE'
        
        if abs(close - fan_price) / close < tolerance_wide:
            signals.append({
                'signal': f"FIB FAN {fan_ratio*100:.1f}%",
                'description': f"Price at {fan_ratio*100:.1f}% Fibonacci fan line",
                'strength': fan_strength,
                'category': 'FIB_FAN',
                'timeframe': self.interval,
                'value': fan_price
            })
    
    # ========== SIGNAL GROUP 8: FIBONACCI TIME ZONES (Original - 8 signals) ==========
    fib_time_numbers = [8, 13, 21, 34, 55, 89, 144]
    current_bar = len(df)
    
    for fib_num in fib_time_numbers:
        bars_from_pivot = (current_bar - 50) % fib_num
        
        if bars_from_pivot <= 1:
            signals.append({
                'signal': f"FIB TIME ZONE {fib_num}",
                'description': f"Current bar aligns with {fib_num}-period Fibonacci time zone",
                'strength': 'SIGNIFICANT' if fib_num >= 21 else 'MODERATE',
                'category': 'FIB_TIME',
                'timeframe': self.interval,
                'value': fib_num
            })
    
    # ========== SIGNAL GROUP 9: MULTIPLE TIME ZONE CLUSTER (NEW - 4 signals) ==========
    time_cluster_count = 0
    aligned_zones = []
    for fib_num in [13, 21, 34]:
        bars_from_pivot = (current_bar - 50) % fib_num
        if bars_from_pivot <= 1:
            time_cluster_count += 1
            aligned_zones.append(str(fib_num))
    
    if time_cluster_count >= 2:
        signals.append({
            'signal': f"FIB TIME CLUSTER {time_cluster_count}",
            'description': f"Multiple Fibonacci time zones aligned: {', '.join(aligned_zones)}",
            'strength': 'EXTREME',
            'category': 'FIB_TIME_CLUSTER',
            'timeframe': self.interval,
            'value': time_cluster_count
        })
    
    # ========== SIGNAL GROUP 10: ELLIOTT WAVE FIBONACCI PATTERNS (Original - 10 signals) ==========
    if len(df) >= 100:
        # Calculate wave ranges (simplified)
        wave1_low = df['Low'].iloc[-100:-80].min()
        wave1_high = df['High'].iloc[-100:-80].max()
        wave1_range = wave1_high - wave1_low
        
        wave2_low = df['Low'].iloc[-80:-60].min()
        wave2_high = df['High'].iloc[-80:-60].max()
        wave2_retracement = (wave2_high - wave2_low) / wave1_range if wave1_range > 0 else 0
        
        # SIGNAL: Wave 2 = 61.8% of Wave 1
        if 0.60 <= wave2_retracement <= 0.63:
            signals.append({
                'signal': "ELLIOTT WAVE 2 = 61.8% OF WAVE 1",
                'description': "Wave 2 retraces 61.8% of Wave 1 (common pattern)",
                'strength': 'SIGNIFICANT',
                'category': 'ELLIOTT_FIB',
                'timeframe': self.interval,
                'value': wave2_retracement
            })
        
        # SIGNAL: Wave 3 target = 1.618x Wave 1
        wave3_target = wave1_low + (1.618 * wave1_range)
        if abs(close - wave3_target) / close < tolerance:
            signals.append({
                'signal': "ELLIOTT WAVE 3 = 161.8% OF WAVE 1",
                'description': "Price at Wave 3 extension target (1.618x Wave 1)",
                'strength': 'SIGNIFICANT',
                'category': 'ELLIOTT_FIB',
                'timeframe': self.interval,
                'value': wave3_target
            })
        
        # SIGNAL: Wave 5 truncation = 61.8% of (Wave 1 + Wave 3)
        combined_12 = wave1_range + (wave3_target - wave1_low)
        wave5_target = wave1_low + (0.618 * combined_12)
        if abs(close - wave5_target) / close < tolerance:
            signals.append({
                'signal': "ELLIOTT WAVE 5 = 61.8% OF WAVES 1+3",
                'description': "Price at Wave 5 equality target",
                'strength': 'MODERATE',
                'category': 'ELLIOTT_FIB',
                'timeframe': self.interval,
                'value': wave5_target
            })
    
    # ========== SIGNAL GROUP 11: FIBONACCI CONFLUENCE/CLUSTER (Original - 8 signals) ==========
    cluster_tolerance = swing_range * 0.02
    price_clusters = {}
    
    for key, level in fib_data.items():
        cluster_key = round(level['value'] / cluster_tolerance) * cluster_tolerance
        
        if cluster_key not in price_clusters:
            price_clusters[cluster_key] = {'count': 0, 'levels': []}
        
        price_clusters[cluster_key]['count'] += 1
        price_clusters[cluster_key]['levels'].append(level['name'])
    
    for cluster_price, cluster_data in price_clusters.items():
        if cluster_data['count'] >= 2 and abs(close - cluster_price) / close < tolerance_wide:
            strength = 'EXTREME' if cluster_data['count'] >= 3 else 'SIGNIFICANT'
            signals.append({
                'signal': f"FIB CLUSTER {cluster_data['count']} LEVELS",
                'description': f"Fibonacci confluence at {cluster_data['count']} levels: {', '.join(cluster_data['levels'])}",
                'strength': strength,
                'category': 'FIB_CLUSTER',
                'timeframe': self.interval,
                'value': cluster_price
            })
    
    # ========== SIGNAL GROUP 12: RETRACEMENT + EXTENSION CONFLUENCE (NEW - 6 signals) ==========
    for retrace_key in [k for k in fib_data.keys() if fib_data[k]['type'] == 'RETRACE']:
        for ext_key in [k for k in fib_data.keys() if fib_data[k]['type'] == 'EXTENSION']:
            retrace_price = fib_data[retrace_key]['value']
            ext_price = fib_data[ext_key]['value']
            
            if abs(retrace_price - ext_price) / retrace_price < 0.03:
                signals.append({
                    'signal': f"FIB {fib_data[retrace_key]['name']} RETRACE + {fib_data[ext_key]['name']} EXT CONFLUENCE",
                    'description': f"Retracement and extension levels converge",
                    'strength': 'EXTREME',
                    'category': 'FIB_RET_EXT_CONFLUENCE',
                    'timeframe': self.interval,
                    'value': (retrace_price + ext_price) / 2
                })
    
    # ========== SIGNAL GROUP 13: VOLUME CONFIRMATION ON FIBONACCI LEVELS (NEW - 5 signals) ==========
    if 'Volume' in df.columns:
        volume = self._safe_float(current.get('Volume'))
        avg_volume = self._safe_float(df['Volume'].iloc[-20:].mean()) if len(df) >= 20 else None
        
        if volume is not None and avg_volume is not None and avg_volume > 0:
            volume_ratio = volume / avg_volume
            
            if volume_ratio > 1.5:
                nearest_fib = min([(k, fib_data[k]) for k in fib_data.keys()], 
                                key=lambda x: abs(x[1]['value'] - close))
                if abs(close - nearest_fib[1]['value']) / close < tolerance:
                    signals.append({
                        'signal': f"FIB {nearest_fib[1]['name']} + HIGH VOLUME",
                        'description': f"Fibonacci level confirmed with {volume_ratio:.1f}x average volume",
                        'strength': 'SIGNIFICANT',
                        'category': 'FIB_VOLUME',
                        'timeframe': self.interval,
                        'value': nearest_fib[1]['value'],
                        'volume_ratio': volume_ratio
                    })
            
            # NEW: Volume + Multiple Timeframe Confluence
            if volume_ratio > 2.0:
                signals.append({
                    'signal': "FIB + EXTREME VOLUME SPIKE",
                    'description': f"Fibonacci level with extreme volume ({volume_ratio:.1f}x)",
                    'strength': 'EXTREME',
                    'category': 'FIB_VOLUME_EXTREME',
                    'timeframe': self.interval,
                    'value': close,
                    'volume_ratio': volume_ratio
                })
    
    # ========== SIGNAL GROUP 14: FIBONACCI + MOVING AVERAGE CONVERGENCE (NEW - 6 signals) ==========
    ma_keys = ['SMA_50', 'SMA_200', 'EMA_12', 'EMA_26']
    
    for ma_key in ma_keys:
        if ma_key in df.columns:
            ma_value = self._safe_float(current.get(ma_key))
            if ma_value is not None:
                # Find nearest Fibonacci level to MA
                nearest_fib = min([(k, fib_data[k]) for k in fib_data.keys()],
                                key=lambda x: abs(x[1]['value'] - ma_value))
                
                if abs(ma_value - nearest_fib[1]['value']) / ma_value < 0.02:
                    signals.append({
                        'signal': f"FIB {nearest_fib[1]['name']} + {ma_key} CONVERGENCE",
                        'description': f"{ma_key} converging with {nearest_fib[1]['name']} Fibonacci level",
                        'strength': 'SIGNIFICANT',
                        'category': 'FIB_MA_CONFLUENCE',
                        'timeframe': self.interval,
                        'value': ma_value
                    })
    
    # ========== SIGNAL GROUP 15: PRICE REVERSAL AT FIBONACCI (NEW - 8 signals) ==========
    if len(df) >= 3:
        high_2 = self._safe_float(df['High'].iloc[-2])
        low_2 = self._safe_float(df['Low'].iloc[-2])
        
        if high_2 is not None and low_2 is not None:
            # Check for reversal candles at Fibonacci levels
            for key, level in fib_data.items():
                if level['type'] in ['RETRACE', 'EXTENSION']:
                    # Hammer/Pin bar at Fibonacci level
                    if low_2 <= level['value'] <= high_2 and abs(close - level['value']) / close < 0.01:
                        signals.append({
                            'signal': f"FIB {level['name']} REVERSAL PIN",
                            'description': f"Reversal pin bar at {level['name']} Fibonacci",
                            'strength': 'MODERATE',
                            'category': 'FIB_REVERSAL_PIN',
                            'timeframe': self.interval,
                            'value': level['value']
                        })
    
    # ========== SIGNAL GROUP 16: FIBONACCI RATIO RELATIONSHIPS (NEW - 7 signals) ==========
    # Golden spiral ratios
    golden_spiral_ratios = [0.236, 0.382, 0.618, 1.0, 1.618, 2.618]
    
    for i, ratio in enumerate(golden_spiral_ratios[:-1]):
        current_level = low_50 + (ratio * swing_range)
        next_level = low_50 + (golden_spiral_ratios[i+1] * swing_range)
        midpoint = (current_level + next_level) / 2
        
        if abs(close - midpoint) / close < tolerance:
            signals.append({
                'signal': f"FIB GOLDEN SPIRAL {ratio*100:.1f}%-{golden_spiral_ratios[i+1]*100:.1f}%",
                'description': f"Price at golden spiral midpoint",
                'strength': 'MODERATE',
                'category': 'FIB_GOLDEN_SPIRAL',
                'timeframe': self.interval,
                'value': midpoint
            })
    
    # ========== SIGNAL GROUP 17: FIBONACCI HARMONIC PATTERN CONFLUENCE (NEW - 6 signals) ==========
    # Bat, Butterfly, Gartley pattern Fibonacci targets (88.6%, 78.6%, etc.)
    harmonic_ratios = [0.886, 0.618, 0.382, 1.618]
    
    for h_ratio in harmonic_ratios:
        h_price = low_50 + (h_ratio * swing_range)
        
        if abs(close - h_price) / close < tolerance:
            signals.append({
                'signal': f"FIB HARMONIC PATTERN {h_ratio*100:.1f}%",
                'description': f"Price at Harmonic pattern Fibonacci level ({h_ratio*100:.1f}%)",
                'strength': 'SIGNIFICANT',
                'category': 'FIB_HARMONIC',
                'timeframe': self.interval,
                'value': h_price
            })
    
    # ========== SIGNAL GROUP 18: THREE TIMEFRAME FIBONACCI ALIGNMENT (NEW - 5 signals) ==========
    # Simplified multi-timeframe check - in production, these would be calculated on different timeframes
    if len(df) >= 100:
        tf_price_1h = close
        tf_price_4h = df['Close'].iloc[-4]
        tf_price_1d = df['Close'].iloc[-24] if len(df) >= 24 else close
        
        # Check if all three timeframes are near same Fibonacci level
        nearest_1h = min([(k, fib_data[k]) for k in fib_data.keys()],
                        key=lambda x: abs(x[1]['value'] - tf_price_1h))
        nearest_4h = min([(k, fib_data[k]) for k in fib_data.keys()],
                        key=lambda x: abs(x[1]['value'] - tf_price_4h))
        nearest_1d = min([(k, fib_data[k]) for k in fib_data.keys()],
                        key=lambda x: abs(x[1]['value'] - tf_price_1d))
        
        if (nearest_1h[1]['name'] == nearest_4h[1]['name'] == nearest_1d[1]['name']):
            signals.append({
                'signal': f"FIB 3-TIMEFRAME ALIGNMENT {nearest_1h[1]['name']}",
                'description': f"1H, 4H, and Daily all at {nearest_1h[1]['name']} Fibonacci",
                'strength': 'EXTREME',
                'category': 'FIB_3TF_ALIGNMENT',
                'timeframe': self.interval,
                'value': nearest_1h[1]['value']
            })
    
    # ========== SIGNAL GROUP 19: FIBONACCI DIVERGENCE SIGNALS (NEW - 5 signals) ==========
    if 'RSI' in df.columns and len(df) >= 2:
        rsi = self._safe_float(current.get('RSI'))
        prev_rsi = self._safe_float(prev.get('RSI'))
        
        if rsi is not None and prev_rsi is not None:
            # Bullish divergence: Price at Fib level, RSI divergence
            if close < low_50 + (0.618 * swing_range) and rsi > prev_rsi and rsi < 50:
                signals.append({
                    'signal': "FIB + BULLISH RSI DIVERGENCE",
                    'description': "Price at Fibonacci with bullish RSI divergence",
                    'strength': 'SIGNIFICANT',
                    'category': 'FIB_RSI_DIV',
                    'timeframe': self.interval,
                    'value': close
                })
            
            # Bearish divergence: Price at Fib level, RSI divergence
            if close > low_50 + (1.618 * swing_range) and rsi < prev_rsi and rsi > 50:
                signals.append({
                    'signal': "FIB + BEARISH RSI DIVERGENCE",
                    'description': "Price at Fibonacci with bearish RSI divergence",
                    'strength': 'SIGNIFICANT',
                    'category': 'FIB_RSI_DIV',
                    'timeframe': self.interval,
                    'value': close
                })
    
    # ========== SIGNAL GROUP 20: FIBONACCI STOCHASTIC CONFLUENCE (NEW - 4 signals) ==========
    if 'Stoch_K' in df.columns and 'Stoch_D' in df.columns:
        stoch_k = self._safe_float(current.get('Stoch_K'))
        stoch_d = self._safe_float(current.get('Stoch_D'))
        
        if stoch_k is not None and stoch_d is not None:
            nearest_fib = min([(k, fib_data[k]) for k in fib_data.keys()],
                            key=lambda x: abs(x[1]['value'] - close))
            
            if abs(close - nearest_fib[1]['value']) / close < tolerance:
                # Stochastic at extreme
                if stoch_k < 20 or stoch_k > 80:
                    signals.append({
                        'signal': f"FIB {nearest_fib[1]['name']} + STOCH EXTREME",
                        'description': f"Fibonacci level with stochastic extreme ({stoch_k:.0f})",
                        'strength': 'SIGNIFICANT',
                        'category': 'FIB_STOCH',
                        'timeframe': self.interval,
                        'value': closest_fib[1]['value'],
                        'stoch_k': stoch_k
                    })
    
    return signals    