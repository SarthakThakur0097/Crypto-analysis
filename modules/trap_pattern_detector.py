# modules/trap_pattern_detector.py

import pandas as pd
import numpy as np
import mplfinance as mpf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates

def flag_bait_and_trap(
    df: pd.DataFrame,
    lookahead: int = 15
) -> pd.DataFrame:
    """
    Flags “bait‐and‐trap” patterns:
      1) fake_break_high: bar’s high > range_top but close ≤ range_top
      2) in next `lookahead` bars:
         a) touched support (low ≤ range_bot)
         b) fake_break_low: low < range_bot & close ≥ range_bot
         c) reversal candle (e.g. engulfing_flag)

    Outputs new columns:
      - fake_break_high (bool)
      - fake_break_low  (bool)
      - bait_trap_pattern (bool)
      - time_to_support   (int bars until first support touch)
    """
    df = df.copy()
    df['fake_break_high']   = (
        (df['high'] > df['range_top']) &
        (df['close'] <= df['range_top'])
    )
    df['fake_break_low']    = False
    df['bait_trap_pattern'] = False
    df['time_to_support']   = np.nan

    for ts in df.index[df['fake_break_high']]:
        pos    = df.index.get_loc(ts)
        window = df.iloc[pos+1 : pos+1+lookahead]

        touched = window['low'] <= df.at[ts, 'range_bot']
        fake_low = (
            (window['low'] < df.at[ts, 'range_bot']) &
            (window['close'] >= df.at[ts, 'range_bot'])
        )
        # if you have candle‐structure flags, include them:
        rev = (
            window.get('engulfing_flag', False) |
            window.get('strong_close', False)
        )

        if touched.any():
            first_loc = touched.idxmax()
            df.at[ts, 'time_to_support'] = window.index.get_loc(first_loc)
        if touched.any() and fake_low.any() and rev.any():
            df.at[ts, 'fake_break_low']    = True
            df.at[ts, 'bait_trap_pattern'] = True

    return df

class TrapPatternConfig:
    def __init__(
        self,
        impulse_min_gain_pct=7.0,
        impulse_max_duration_bars=70,
        impulse_bos_lookback=10,
        range_min_bars=60,
        range_max_range_pct=5.0,
        range_min_touch_count=2,
        range_top_margin_pct=1.0
    ):
        self.impulse_min_gain_pct = impulse_min_gain_pct
        self.impulse_max_duration_bars = impulse_max_duration_bars
        self.impulse_bos_lookback = impulse_bos_lookback

        self.range_min_bars = range_min_bars
        self.range_max_range_pct = range_max_range_pct
        self.range_min_touch_count = range_min_touch_count
        self.range_top_margin_pct = range_top_margin_pct


def detect_impulses(df: pd.DataFrame, config: TrapPatternConfig):
    impulses = []
    for i in range(len(df) - config.impulse_max_duration_bars):
        window = df.iloc[i:i + config.impulse_max_duration_bars]
        start_price = window.iloc[0]['close']

        price_diff = window['close'].max() - start_price
        gain_pct = (price_diff / start_price) * 100
        if gain_pct < config.impulse_min_gain_pct:
            continue

        pivot_check_window = df.iloc[max(i - config.impulse_bos_lookback, 0):i]
        recent_high = pivot_check_window['high'].max() if not pivot_check_window.empty else -np.inf
        if window['high'].max() <= recent_high:
            continue

        impulse_end_idx = window['close'].idxmax()
        impulse_info = {
            'start_idx': df.index[i],
            'end_idx': impulse_end_idx,
            'start_price': start_price,
            'end_price': df.loc[impulse_end_idx]['close'],
            'gain_pct': gain_pct
        }
        impulses.append(impulse_info)

    return impulses

def find_anchor_support(df, impulse_end_idx, impulse_start_idx):
    impulse_window = df.loc[impulse_start_idx:impulse_end_idx]
    prices = impulse_window['high'].values
    for i in range(1, len(prices) - 1):
        if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
            retrace = (prices[i] - impulse_window['low'].iloc[i+1]) / prices[i]
            if retrace >= 0.015:
                return impulse_window.index[i], prices[i]
    return impulse_window.index[-1], impulse_window['high'].iloc[-1]

def select_best_support(range_window, anchor_support):
    lows = range_window['low'].round(1)
    candidate_levels = lows.value_counts().to_dict()

    best_score = -np.inf
    best_level = None

    for level, count in candidate_levels.items():
        close_above = (range_window['close'] >= level).sum()
        dist_to_anchor = abs(level - anchor_support)

        score = (
            (1 / (dist_to_anchor + 1e-5)) * 0.4 +
            count * 0.3 +
            close_above * 0.2 -
            range_window[range_window['low'].round(1) == level]['low'].std() * 0.1
        )

        if score > best_score:
            best_score = score
            best_level = level

    return best_level, best_score

def detect_consolidations(df: pd.DataFrame, impulses: list, config: TrapPatternConfig):
    ranges = []

    for impulse in impulses:
        end_loc = df.index.get_loc(impulse['end_idx'])
        range_window = df.iloc[end_loc + 1:end_loc + 1 + config.range_min_bars]

        if len(range_window) < config.range_min_bars:
            continue

        high = range_window['high'].max()
        low = range_window['low'].min()
        range_pct = (high - low) / low * 100

        if range_pct > config.range_max_range_pct:
            continue

        top_touches = ((range_window['high'] >= high * (1 - config.range_top_margin_pct / 100)).sum())
        bot_touches = ((range_window['low'] <= low * (1 + config.range_top_margin_pct / 100)).sum())

        # ---- Hybrid validation ----
        if top_touches >= config.range_min_touch_count and bot_touches >= config.range_min_touch_count:
            range_type = 'multi-touch'
        else:
            vol_range_pct = range_pct
            closes = range_window['close']
            low_q = closes.quantile(0.2)
            high_q = closes.quantile(0.8)
            clustered = closes[(closes >= low_q) & (closes <= high_q)]

            if vol_range_pct <= 4 and len(clustered) / len(closes) >= 0.7:
                range_type = 'clustered'
            else:
                continue  # Reject range if neither condition passes

        # Get anchor and behavioral support
        anchor_idx, anchor_support = find_anchor_support(df, impulse['end_idx'], impulse['start_idx'])
        best_support, support_score = select_best_support(range_window, anchor_support)

        ranges.append({
            'range_start': range_window.index[0],
            'range_end': range_window.index[-1],
            'range_high': high,
            'range_low': low,
            'impulse_end': impulse['end_idx'],
            'anchor_support': anchor_support,
            'best_support': best_support,
            'support_score': support_score,
            'type': range_type
        })

    return ranges






def visualize_trap_context(
    df,
    ranges,
    start_date,
    end_date,
    show_ranges=True,
    show_impulses=True
):
    df = df.copy()
    df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
    if df.empty:
        print("No data in selected date range.")
        return

    ohlc = df[['open', 'high', 'low', 'close', 'volume']].copy()
    ohlc.index.name = 'Date'

    red_lines = set()
    apds = []

    for r in ranges:
        if not (r["range_start"] >= pd.to_datetime(start_date) and r["range_end"] <= pd.to_datetime(end_date)):
            continue

        red_lines.add(round(r["range_high"], 2))
        red_lines.add(round(r["range_low"], 2))
        red_lines.add(round(r["best_support"], 2))

        if show_impulses:
            try:
                idx = ohlc.index.get_loc(r['impulse_end'])
                marker = np.full(len(ohlc), np.nan)
                marker[idx] = ohlc['low'].iloc[idx]
                apds.append(mpf.make_addplot(marker, type='scatter', markersize=60, marker='^', color='blue'))
            except:
                continue

    mpf.plot(
        ohlc,
        type='candle',
        style='yahoo',
        volume=True,
        addplot=apds,
        hlines=sorted(red_lines),
        title=f"Trap Context Visualization: {start_date} to {end_date}",
        figratio=(16, 8),
        figscale=1.4
    )

# Example usage:
# config = TrapPatternConfig()
# impulses = detect_impulses(df, config)
# ranges = detect_consolidations(df, impulses, config)
# visualize_trap_context(df, ranges, "2025-04-21", "2025-05-02")
