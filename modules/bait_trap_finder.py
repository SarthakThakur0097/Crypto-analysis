# modules/bait_trap_finder.py

import pandas as pd
from typing import List, Optional
from modules.base_zone_detector import BaseZoneDetector

class BaitTrapFinder:
    """
    Detects “equal‑low → fake‑break → re‑entry → 1.618 target” patterns.

    Parameters
    ----------
    lookback_bars : int
        How many bars back to use for pivot detection.
    equal_low_tol : float
        Tolerance for considering two lows “equal” (e.g. 0.001 = 0.1%).
    min_pivots : int
        Minimum number of touches at the base_low to qualify.
    fake_break_tol : float
        How far above base_high the high must extend (e.g. 0.001 = 0.1%).
    reentry_buffer : float
        How deep into the base_low zone the candle must close 
        (e.g. 0.002 = 0.2% above the low).
    max_pullback_bars : int
        How many bars after the base end to search for the trap bar
        and then for the 1.618 target.
    fib_multiplier : float
        Which Fibonacci extension to project (default 1.618).
    """
    def __init__(
        self,
        lookback_bars:      int,
        equal_low_tol:      float,
        min_pivots:         int,
        fake_break_tol:     float,
        reentry_buffer:     float,
        max_pullback_bars:  int,
        fib_multiplier:     float = 1.618
    ):
        self.lookback_bars     = lookback_bars
        self.equal_low_tol     = equal_low_tol
        self.min_pivots        = min_pivots
        self.fake_break_tol    = fake_break_tol
        self.reentry_buffer    = reentry_buffer
        self.max_pullback_bars = max_pullback_bars
        self.fib_multiplier    = fib_multiplier

    def find_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scan the entire DataFrame for bait‑and‑trap setups.

        Returns a DataFrame with one row per detected pattern, columns:
          ['pattern_id','base_low','base_high',
           'start_ts','end_ts','trap_ts',
           'fib_target','hit_target','bars_to_target','target_ts']
        """
        # ensure datetime index sorted
        df = df.sort_index()
        n = len(df)

        # Step 1: detect all equal‑low bases
        bzd = BaseZoneDetector(
            lookback_bars=self.lookback_bars,
            equal_low_tol=self.equal_low_tol,
            min_pivots=self.min_pivots
        )
        bases = bzd.detect_bases(df)

        records: List[dict] = []

        # Step 2: for each base, look for the fake‑break + trap bar
        for pid, base in enumerate(bases):
            low       = base['base_low']
            high      = base['base_high']
            start_idx = base['start_idx']
            end_idx   = base['end_idx']
            start_ts  = df.index[start_idx]
            end_ts    = df.index[end_idx]

            trap_idx: Optional[int] = None
            # scan forward for trap bar
            for look_idx in range(end_idx+1,
                                  min(n, end_idx+1 + self.max_pullback_bars)):
                row = df.iloc[look_idx]
                # fake‑break above high
                broke_high = row['high'] > high * (1 + self.fake_break_tol)
                # close back inside the box
                closed_inside = (
                    (row['close'] <= high) and
                    (row['close'] >= low * (1 + self.reentry_buffer))
                )
                # dipped down to low zone
                dipped_low = row['low'] <= low * (1 + self.reentry_buffer)

                if broke_high and closed_inside and dipped_low:
                    trap_idx = look_idx
                    break

            if trap_idx is None:
                # no trap found for this base
                continue

            trap_ts   = df.index[trap_idx]
            # Step 3: compute fib target
            fib_target = low + (high - low) * self.fib_multiplier

            # Step 4: scan for the first hit of that target
            hit_idx: Optional[int] = None
            for hit_look in range(trap_idx+1,
                                  min(n, trap_idx+1 + self.max_pullback_bars)):
                if df.iloc[hit_look]['high'] >= fib_target:
                    hit_idx = hit_look
                    break

            hit_target     = hit_idx is not None
            bars_to_target = (hit_idx - trap_idx) if hit_target else None
            target_ts      = df.index[hit_idx] if hit_target else None

            records.append({
                'pattern_id':    pid,
                'base_low':      low,
                'base_high':     high,
                'start_ts':      start_ts,
                'end_ts':        end_ts,
                'trap_ts':       trap_ts,
                'fib_target':    fib_target,
                'hit_target':    hit_target,
                'bars_to_target':bars_to_target,
                'target_ts':     target_ts
            })

        return pd.DataFrame.from_records(records)
