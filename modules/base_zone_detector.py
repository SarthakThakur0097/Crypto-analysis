# modules/base_zone_detector.py

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import List, Dict, Optional

class BaseZoneDetector:
    def __init__(
        self,
        lookback_bars:  int     = 12,       # how many bars to look back for pivots
        equal_low_tol:  float   = 0.001,    # ±0.1% tolerance on lows
        min_pivots:     int     = 2,        # need at least this many equal‑low pivots
        min_span:       Optional[int]   = None,  # min number of bars in the base
        max_span:       Optional[int]   = None,  # max number of bars in the base
        max_height_atr: Optional[float] = None   # max (height / ATR) allowed
    ):
        self.lookback_bars  = lookback_bars
        self.equal_low_tol  = equal_low_tol
        self.min_pivots     = min_pivots
        self.min_span       = min_span
        self.max_span       = max_span
        self.max_height_atr = max_height_atr

    def detect_bases(self, df: pd.DataFrame) -> List[Dict]:
        """
        Scan df for multi‑pivot “bases” of equal lows.
        Returns a list of dicts, each containing:
          - base_low:   average low of the clustered pivots
          - base_high:  max high over that pivot span
          - start_idx:  integer position of first pivot
          - end_idx:    integer position of last pivot
        Applies optional span & height/ATR filters if configured.
        """
        lows = df['low'].values
        # 1) find local minima (pivots)
        pivots = argrelextrema(lows, np.less_equal, order=self.lookback_bars)[0]

        # 2) cluster pivots within tolerance
        raw_bases = []
        for pivot in pivots:
            this_low = lows[pivot]
            close_pivots = [
                p for p in pivots
                if abs(lows[p] - this_low) / this_low <= self.equal_low_tol
            ]
            if len(close_pivots) >= self.min_pivots:
                start, end = min(close_pivots), max(close_pivots)
                base_low  = float(np.mean(lows[close_pivots]))
                base_high = float(df['high'].iloc[start:end+1].max())
                raw_bases.append({
                    'base_low':  base_low,
                    'base_high': base_high,
                    'start_idx': start,
                    'end_idx':   end
                })

        # 3) dedupe overlapping clusters
        unique = []
        for b in raw_bases:
            if not any(
                (b['start_idx'] <= u['end_idx'] and b['end_idx'] >= u['start_idx'])
                for u in unique
            ):
                unique.append(b)

        # 4) apply optional filters
        filtered = []
        for b in unique:
            span   = b['end_idx'] - b['start_idx'] + 1
            height = b['base_high'] - b['base_low']

            if self.min_span and span < self.min_span:
                continue
            if self.max_span and span > self.max_span:
                continue
            if self.max_height_atr is not None:
                # need ATR_14 present in df
                atr_series = df['ATR_14'].iloc[b['start_idx']:b['end_idx']+1]
                avg_atr = atr_series.mean() if not atr_series.isna().all() else np.nan
                if np.isnan(avg_atr) or (height / avg_atr) > self.max_height_atr:
                    continue

            filtered.append(b)

        return filtered
