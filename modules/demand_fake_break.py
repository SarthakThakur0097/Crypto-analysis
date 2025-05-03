# modules/demand_fake_break.py

import pandas as pd
from typing import List, Dict
from .fib_targets import compute_extensions

class DemandFakeBreakDetector:
    def __init__(
        self,
        pullback_window: int   = 12,    # bars after base to catch the pullback
        buffer_tol:      float = 0.001  # 0.1% leeway around base low
    ):
        self.pullback_window = pullback_window
        self.buffer_tol      = buffer_tol

    def flag_patterns(
        self,
        df: pd.DataFrame,
        bases: List[Dict]
    ) -> pd.DataFrame:
        """
        Given a df with OHLC and a list of base dicts (from BaseZoneDetector),
        find the first bar after each base_high that:
          a) dips low ≤ base_low*(1+buffer_tol)  (bait)
          b) closes back ≥ base_low                (trap)
        Attach to df:
          - pattern_id (int)
          - base_low, base_high
          - trap_entry (bool)
          - fib_target_1.0, fib_target_1.618
        """
        df = df.copy()
        # initialize columns
        df['pattern_id']      = pd.NA
        df['trap_entry']      = False
        df['base_low']        = pd.NA
        df['base_high']       = pd.NA
        df['fib_target_1.0']  = pd.NA
        df['fib_target_1.618']= pd.NA

        for pid, base in enumerate(bases):
            e = base['end_idx']
            low, high = base['base_low'], base['base_high']
            # find first bar after e that breaks above the base_high
            post = df.iloc[e+1 : e+1+self.pullback_window]
            trigger_idx = post.index[post['high'] > high]
            if trigger_idx.empty:
                continue
            entry_bar = trigger_idx[0]

            # in the following pullback_window, find first low ≤ low*(1+tol)
            pb = df.loc[entry_bar:].iloc[:self.pullback_window]
            bait_mask = pb['low'] <= low*(1 + self.buffer_tol)
            if not bait_mask.any():
                continue
            bait_idx = bait_mask.idxmax()

            # ensure that same bar’s close ≥ base_low
            if df.at[bait_idx, 'close'] < low:
                continue

            # flag the pattern
            df.at[bait_idx, 'pattern_id'] = pid
            df.at[bait_idx, 'trap_entry'] = True
            df.at[bait_idx, 'base_low']   = low
            df.at[bait_idx, 'base_high']  = high

            # compute extensions
            exts = compute_extensions(low, high, ratios=[1.0, 1.618])
            df.at[bait_idx, 'fib_target_1.0']   = exts['target_1.0']
            df.at[bait_idx, 'fib_target_1.618'] = exts['target_1.618']

        return df
