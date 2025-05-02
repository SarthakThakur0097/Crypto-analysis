# modules/base_zone_detector.py

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import List, Dict

class BaseZoneDetector:
    def __init__(
        self,
        lookback_bars: int    = 12,      # how many bars to look for a base
        equal_low_tol: float  = 0.001,   # ±0.1% tolerance on lows
        min_pivots:     int   = 2        # at least this many equal‑low pivots
    ):
        self.lookback_bars  = lookback_bars
        self.equal_low_tol  = equal_low_tol
        self.min_pivots     = min_pivots

    def detect_bases(self, df: pd.DataFrame) -> List[Dict]:
        """
        Scan the entire df for base zones,
        returning a list of dicts:
        {
           'base_low': float,
           'base_high': float,
           'start_idx': int,     # integer position in df
           'end_idx':   int      # integer position in df
        }
        """
        prices = df['low'].values
        # find local minima
        pivots = argrelextrema(prices, np.less_equal, order=self.lookback_bars)[0]

        bases = []
        for i in range(len(pivots)):
            # collect pivots within tolerance of this pivot’s price
            low_val = prices[pivots[i]]
            close_idxs = [
                j for j in pivots
                if abs(prices[j] - low_val) / low_val <= self.equal_low_tol
            ]
            if len(close_idxs) >= self.min_pivots:
                start, end = min(close_idxs), max(close_idxs)
                base_low  = float(np.mean(prices[close_idxs]))
                base_high = float(df['high'].iloc[start:(end+1)].max())
                bases.append({
                    'base_low':  base_low,
                    'base_high': base_high,
                    'start_idx': start,
                    'end_idx':   end
                })
        # remove duplicates (clusters that overlap)
        unique = []
        for b in bases:
            if not any(
                (b['start_idx'] <= u['end_idx'] and b['end_idx'] >= u['start_idx'])
                for u in unique
            ):
                unique.append(b)
        return unique
