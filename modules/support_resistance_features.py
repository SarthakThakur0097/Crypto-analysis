# modules/support_resistance_features.py

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN

def calculate_pivots(df: pd.DataFrame, order: int = 5) -> pd.DataFrame:
    """
    Identify local pivot highs & lows using a window of `order` bars.
    """
    df = df.copy()
    highs = df['high'].values
    lows  = df['low'].values

    piv_hi_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
    piv_lo_idx = argrelextrema(lows,  np.less_equal,    order=order)[0]

    df['pivot_high'] = False
    df['pivot_low']  = False
    df.iloc[piv_hi_idx, df.columns.get_loc('pivot_high')] = True
    df.iloc[piv_lo_idx, df.columns.get_loc('pivot_low')]  = True
    return df

def cluster_zones(
    prices: np.ndarray,
    eps: float = 0.002,
    min_samples: int = 2
) -> list[dict]:
    """
    Cluster pivot prices into discrete zones using DBSCAN.
    eps is relative (fractional) scale of price.
    """
    if prices.size == 0:
        return []
    pts = prices.reshape(-1,1)
    # scale eps by average price
    db = DBSCAN(eps=eps * prices.mean(), min_samples=min_samples).fit(pts)
    zones = []
    for label in sorted(set(db.labels_)):
        if label < 0:
            continue
        cluster = prices[db.labels_ == label]
        zones.append({
            'zone_price': float(cluster.mean()),
            'count':      int(cluster.size),
            'label':      int(label)
        })
    return zones

def detect_zones(
    df: pd.DataFrame,
    pivot_order:     int,
    zone_eps:        float,
    min_zone_points: int
) -> tuple[list[dict], list[dict]]:
    """
    Return (support_zones, resistance_zones) as lists of dicts.
    """
    df_piv = calculate_pivots(df, order=pivot_order)
    highs = df_piv.loc[df_piv.pivot_high, 'high'].to_numpy()
    lows  = df_piv.loc[df_piv.pivot_low,  'low' ].to_numpy()
    res_zones = cluster_zones(highs, eps=zone_eps, min_samples=min_zone_points)
    sup_zones = cluster_zones(lows,  eps=zone_eps, min_samples=min_zone_points)
    return sup_zones, res_zones

def label_zone_features(
    df: pd.DataFrame,
    support_zones:    list[dict],
    resistance_zones: list[dict],
    buffer:           float = 0.001
) -> pd.DataFrame:
    """
    For each bar, compute:
      - dist_to_support (fractional)
      - in_support_zone (bool if dist ≤ buffer)
      - dist_to_resistance
      - in_resistance_zone
    """
    df = df.copy()
    sup_prices = np.array([z['zone_price'] for z in support_zones])
    res_prices = np.array([z['zone_price'] for z in resistance_zones])

    df['dist_to_support']    = np.nan
    df['dist_to_resistance'] = np.nan
    df['in_support_zone']    = False
    df['in_resistance_zone'] = False

    for idx, row in df.iterrows():
        low, high = row.low, row.high
        if sup_prices.size:
            d_sup = np.abs(low - sup_prices) / sup_prices
            df.at[idx, 'dist_to_support'] = float(d_sup.min())
            if d_sup.min() <= buffer:
                df.at[idx, 'in_support_zone'] = True

        if res_prices.size:
            d_res = np.abs(high - res_prices) / res_prices
            df.at[idx, 'dist_to_resistance'] = float(d_res.min())
            if d_res.min() <= buffer:
                df.at[idx, 'in_resistance_zone'] = True

    return df

def assign_active_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given df.attrs['support_zones'] & ['resistance_zones'],
    assign each bar its nearest range_bot & range_top.
    """
    df = df.copy()
    sup_list = sorted(z['zone_price'] for z in df.attrs.get('support_zones', []))
    res_list = sorted(z['zone_price'] for z in df.attrs.get('resistance_zones', []))

    df['range_bot'] = np.nan
    df['range_top'] = np.nan

    for idx, row in df.iterrows():
        c = row.close
        # next larger resistance
        tops = [p for p in res_list if p > c]
        df.at[idx, 'range_top'] = tops[0] if tops else np.nan
        # next smaller support
        bots = [p for p in reversed(sup_list) if p < c]
        df.at[idx, 'range_bot'] = bots[0] if bots else np.nan

    return df

def add_support_resistance_features(
    df: pd.DataFrame,
    pivot_order:     int   = 5,
    zone_eps:        float = 0.002,
    min_zone_points: int   = 2,
    buffer:          float = 0.001
) -> pd.DataFrame:
    """
    Master entrypoint: detect pivots, cluster zones, label distances/in-zone,
    and store zone definitions in df.attrs for assign_active_range().
    """
    df = df.copy()
    sup_z, res_z = detect_zones(df, pivot_order, zone_eps, min_zone_points)
    df = label_zone_features(df, sup_z, res_z, buffer)
    df.attrs['support_zones']    = sup_z
    df.attrs['resistance_zones'] = res_z
    return df
